from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch._dynamo
from torch import nn

from .. import fast_uses_compile, is_fast_inference
from ..synaptogen import CellArrayCPU
from .util import poly_mul, poly_mul_horner, randn_broadcast


_COMPILED_FUSED_FORWARD = None


def _fused_memristor_forward(
    low_poly, high_poly, r, inputs, noise_sample, kBT, BW, electron_e, noise_min
):
    """Fused memristor forward shared by all MemristorArray instances.

    Algebraically identical to the eager path, evaluated with Horner. The readout
    noise is passed in (drawn eagerly by the caller) so the random draw matches
    eager exactly; only kernel fusion / Horner introduce ~1e-6 drift.

    Defined at module level (not as a bound method) on purpose: torch.compile then
    produces ONE shared graph keyed on tensor shapes, instead of a separate
    compilation per instance (which guards on ``self``'s id). A model with many
    arrays would otherwise blow the dynamo cache and silently fall back to eager.
    """
    result_low = poly_mul_horner(low_poly, inputs).unsqueeze(-1)
    result_high = poly_mul_horner(high_poly, inputs).unsqueeze(-1)
    result_raw = result_low * (1 - r) + result_high * r

    abs_raw = torch.abs(result_raw)
    denom = torch.abs(inputs.unsqueeze(-1)) + noise_min
    johnson_noise = 4 * kBT * BW * (abs_raw / denom)
    shot_noise = 2 * electron_e * abs_raw * BW
    sigma_total = torch.sqrt(johnson_noise + shot_noise)

    result_noised = result_raw + noise_sample * sigma_total
    return torch.sum(result_noised, dim=-2)


def _get_compiled_fused_forward():
    """Lazily build (once) the shared compiled fused forward."""
    global _COMPILED_FUSED_FORWARD
    if _COMPILED_FUSED_FORWARD is None:
        # Only a few input ranks (linear vs conv) compile; give dynamo headroom so
        # they all stay cached rather than falling back to eager.
        if torch._dynamo.config.cache_size_limit < 64:
            torch._dynamo.config.cache_size_limit = 64
        _COMPILED_FUSED_FORWARD = torch.compile(_fused_memristor_forward, dynamic=True)
    return _COMPILED_FUSED_FORWARD


class MemristorArray(nn.Module):
    """
    Torch Module for a Memristor Array.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        low_degree: int = 7,
        high_degree: int = 6,
        *,
        additional_axes: Sequence[int] = (),
        broadcast_noise_dims: int = 1,
    ):
        """
        :param in_features: input lines of the memristor (I)
        :param out_features: output lines of the memristor (O)
        :param low_degree:
        :param high_degree:
        :param additional_axes: additional batch axes
        :param broadcast_noise_dims: number of leading dimensions to broadcast noise over, performance optimization
        """
        super().__init__()

        # the resistance state can be applied to both polynomials beforehand
        self.resistance_weighted_poly_low = nn.Parameter(
            torch.empty((low_degree,)), requires_grad=False
        )
        self.resistance_weighted_poly_high = nn.Parameter(
            torch.empty((high_degree,)), requires_grad=False
        )
        self.r = nn.Parameter(
            torch.empty(additional_axes + (in_features, out_features)),  # input major
            requires_grad=False,
        )
        self.num_additional_axes = len(additional_axes)
        self.low_degree = low_degree
        self.high_degree = high_degree

        self.BW = 1e-8  # Default Constant
        self.kBT = 1.380649e-23 * 300  # Default Constant
        self.noise_minimum_voltage = 1e-12
        self.e = np.exp(1)
        assert broadcast_noise_dims >= 0
        self.broadcast_noise_dims = broadcast_noise_dims

        self.in_features = in_features
        self.out_features = out_features

    def init_resistance_states(self, cells: CellArrayCPU):
        LLRS = torch.Tensor(cells.params.LLRS)
        # internal computations are easier if we start from lowest polynomial first, so flip
        LLRS = torch.flip(LLRS, dims=[0])
        self.resistance_weighted_poly_low.data = LLRS

        HHRS = torch.Tensor(cells.params.HHRS)
        HHRS = torch.flip(HHRS, dims=[0])
        self.resistance_weighted_poly_high.data = HHRS

    def init_from_cell_array_input_major(self, cells: CellArrayCPU):
        self.init_resistance_states(cells)
        self.r.data = torch.Tensor(cells.r).resize(*self.r.shape)

    def compute_raw_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: [..., I]
        :return [..., I, O]
        """
        result_low = poly_mul(self.resistance_weighted_poly_low, inputs).unsqueeze(-1)
        result_high = poly_mul(self.resistance_weighted_poly_high, inputs).unsqueeze(-1)
        # result_low * (1 - r) + result_high * r, accumulated in place to avoid a
        # second full [..., I, O] temporary. Same arithmetic -> bit-identical.
        result_raw = result_low * (1 - self.r)  # [..., ...A, I, O]
        result_raw += result_high * self.r
        return result_raw

    def compute_noise(
        self, result_raw: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """

        :param result_raw: [..., ...A, I, O]
        :param inputs: [..., I]
        :return:
        """
        # abs(result_raw) is needed by both noise terms; compute it once. For the
        # johnson term the denominator is strictly positive, so
        # abs(result_raw / denom) == abs(result_raw) / denom exactly in IEEE.
        abs_raw = torch.abs(result_raw)
        johnson_noise = (
            4
            * self.kBT
            * self.BW
            * (abs_raw / (torch.abs(inputs.unsqueeze(-1)) + self.noise_minimum_voltage))
        )
        shot_noise = 2 * self.e * abs_raw * self.BW
        sigma_total = torch.sqrt(johnson_noise + shot_noise)
        noise = randn_broadcast(
            result_raw.shape, self.broadcast_noise_dims, device=inputs.device
        )
        return noise * sigma_total

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: [...B, I]
        :return: [...B, ...A, O]
        """
        if is_fast_inference():
            return self._forward_fast(inputs)

        result_raw = self.compute_raw_output(inputs)
        noise = self.compute_noise(result_raw, inputs)
        # result_raw is no longer needed separately; add the noise in place to save
        # a full [..., I, O] temporary. Same arithmetic -> bit-identical.
        result_raw += noise

        return torch.sum(
            result_raw, dim=-2
        )  # [...B, ...A, I, O] -> sum reduce I -> [...B, ...A, O]

    # -------------------------------------------------------------- fast path ---
    def _forward_fast(self, inputs: torch.Tensor):
        # Draw the readout noise eagerly with the SAME trailing shape the eager
        # path uses: randn_broadcast samples torch.randn(shape[broadcast_dims:]).
        # Keeping the draw outside the compiled region makes the random numbers
        # identical to eager, so the only difference is the fused arithmetic.
        raw_shape = torch.broadcast_shapes(inputs.shape + (1,), self.r.shape)
        trailing_shape = raw_shape[self.broadcast_noise_dims :]
        noise_sample = torch.randn(trailing_shape, device=inputs.device)

        forward = (
            _get_compiled_fused_forward()
            if fast_uses_compile()
            else _fused_memristor_forward
        )
        return forward(
            self.resistance_weighted_poly_low,
            self.resistance_weighted_poly_high,
            self.r,
            inputs,
            noise_sample,
            self.kBT,
            self.BW,
            self.e,
            self.noise_minimum_voltage,
        )


class PairedMemristorArrayV2(nn.Module):
    """
    TorchModule for a Memristor array with pairwise subtracted bitlines
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pos = MemristorArray(*args, **kwargs)
        self.neg = MemristorArray(*args, **kwargs)

    def init_from_paired_cell_array_input_major(
        self, positive_cells: CellArrayCPU, negative_cells: CellArrayCPU
    ):
        self.pos.init_from_cell_array_input_major(positive_cells)
        self.neg.init_from_cell_array_input_major(negative_cells)

    def forward(self, inputs: torch.Tensor):
        return self.pos.forward(inputs) - self.neg.forward(inputs)


@dataclass
class DacAdcHardwareSettings:
    input_bits: int
    output_precision_bits: int
    output_range_bits: int
    hardware_input_vmax: float
    hardware_output_current_scaling: float

    def __post_init__(self):
        self.dac_input_quant_scaling_factor = 1 / 2 ** (self.input_bits - 1)
        self.dac_max = 2 ** (self.input_bits - 1)
        self.dac_min = -self.dac_max
        self.adc_max = 2 ** (self.output_precision_bits + self.output_range_bits - 1)
        self.adc_min = -self.adc_max


class DacAdcPair(nn.Module):
    """
    Simple variant of an DAC and ADC Module,
    computationally related to "input quantizer" and "output quantizer"
    """

    def __init__(self, hardware_settings: DacAdcHardwareSettings):
        super().__init__()
        """
        :param hardware_settings: Should usually be the same for the whole setup / every instance
        """
        self.hs = hardware_settings

    def dac(self, tensor: torch.Tensor):
        int_quantized_input = torch.fake_quantize_per_tensor_affine(
            tensor,
            scale=self.hs.dac_input_quant_scaling_factor,
            zero_point=0,
            quant_min=self.hs.dac_min,
            quant_max=self.hs.dac_max,
        )

        # * self.HARDWARE_VMAX -> convert to physical voltage
        input_voltage = int_quantized_input * self.hs.hardware_input_vmax
        return input_voltage

    def adc(self, tensor: torch.Tensor):
        # go from physical current to internal value space again
        adc_in = tensor * self.hs.hardware_output_current_scaling
        # the precision should quantize [-1,1], while everything larger should be covered by output bits,
        # so for 5 output bits we are quantizing a range of [-32, 32]
        quantized_output = torch.fake_quantize_per_tensor_affine(
            adc_in,
            scale=1 / (2**self.hs.output_precision_bits),
            zero_point=0,
            quant_min=self.hs.adc_min,
            quant_max=self.hs.adc_max,
        )
        return quantized_output
