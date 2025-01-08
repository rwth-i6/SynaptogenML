from dataclasses import dataclass
from typing import Optional, Tuple

import numpy
import numpy as np
import torch
from torch import nn

from .synaptogen import CellArrayCPU
from .quant_modules import LinearQuant, ActivationQuantizer


def poly_mul(coefficients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a polynomial function on all input elements.

    :param degree: Degree of the polynomial
    :param coefficients: polynomial coefficients of shape [P], in ascending order of degree
    :param inputs: inputs ("x") to be evaluated in the shape of [..., I]
    :return: [...., I] outputs
    """
    exponents = torch.arange(0, coefficients.shape[-1], device=inputs.device)  # P
    inputs = inputs.unsqueeze(-1)  # [..., I, 1]
    result = coefficients * (inputs**exponents)  # [..., I, P]
    return result.sum(dim=-1)  # [..., I]


def poly_mul_horner(coefficients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a polynomial function on all input elements using Horner's optimized method.

    Not sure yet if this is faster than the naive implementation.

    :param degree: Degree of the polynomial
    :param coefficients: polynomial coefficients of shape [P], in ascending order of degree
    :param inputs: inputs ("x") to be evaluated in the shape of [..., I]
    :return: [...., I] outputs
    """
    results = torch.zeros_like(inputs, device=inputs.device)
    for coeff in coefficients.flip(-1):  # highest order first
        results = results * inputs + coeff
    return results


def randn_broadcast(
    shape: Tuple[int],
    num_broadcast_dims: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute gaussian noise for all dims except the leading ``num_batch_dims`` and
    broadcast along these.

    Saves memory and computation time for large batch sizes.

    :param shape: Shape of the tensor to be created, split in [...num_batch_dims, ..dims].
    :param num_broadcast_dims: Number of batch dims across which noise is broadcast.
    :param device: the device on which to create the tensor
    """
    assert len(shape) > num_broadcast_dims
    trailing_shape = shape[num_broadcast_dims:]
    trailing_tensor = torch.randn(trailing_shape, device=device)
    broadcasted_tensor = trailing_tensor.expand(shape)
    return broadcasted_tensor


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
        broadcast_noise_dims: int = 1,
    ):
        """

        :param in_features: input lines of the memristor (I)
        :param out_features: output lines of the memristor (O)
        :param low_degree:
        :param high_degree:
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
            torch.empty((out_features, in_features)), requires_grad=False
        )
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
        self.r.data = torch.Tensor(cells.r).resize(self.in_features, self.out_features)

    def compute_raw_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: [..., I]
        :return [..., I, O]
        """
        result_low = poly_mul(self.resistance_weighted_poly_low, inputs).unsqueeze(-1)
        result_high = poly_mul(self.resistance_weighted_poly_high, inputs).unsqueeze(-1)
        result_raw = (1 - self.r) * result_low + self.r * result_high  # [....., I, O]
        return result_raw

    def compute_noise(
        self, result_raw: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """

        :param result_raw: [..., I, O]
        :param inputs: [..., I]
        :return:
        """
        johnson_noise = (
            4
            * self.kBT
            * self.BW
            * torch.abs(
                result_raw
                / (torch.abs(inputs.unsqueeze(-1)) + self.noise_minimum_voltage)
            )
        )
        shot_noise = 2 * self.e * torch.abs(result_raw) * self.BW
        sigma_total = torch.sqrt(johnson_noise + shot_noise)
        noise = randn_broadcast(
            result_raw.shape, self.broadcast_noise_dims, device=inputs.device
        )
        return noise * sigma_total

    def forward(self, inputs: torch.Tensor):
        result_raw = self.compute_raw_output(inputs)

        noise = self.compute_noise(result_raw, inputs)
        result_noised = result_raw + noise

        return torch.sum(
            result_noised, dim=-2
        )  # [..., I, O] -> sum reduce I -> [..., O]


class PairedMemristorArrayV2(MemristorArray):
    """
    TorchModule for a Memristor array with pairwise subtracted bitlines
    """

    def init_from_cell_array_input_major(self, cells: CellArrayCPU):
        raise NotImplementedError

    def init_from_paired_cell_array_input_major(
        self, positive_cells: CellArrayCPU, negative_cells: CellArrayCPU
    ):
        self.init_resistance_states(positive_cells)

        positive_r = torch.Tensor(positive_cells.r).resize(
            self.in_features, self.out_features
        )
        negative_r = torch.Tensor(negative_cells.r).resize(
            self.in_features, self.out_features
        )
        self.r.data = torch.concat([positive_r, negative_r], dim=-1)  # [I, 2 * O]

    def forward(self, inputs: torch.Tensor):
        result_raw = self.compute_raw_output(inputs)

        # compute gaussian noises
        noise = self.compute_noise(result_raw, inputs)

        # Apply noise
        result_noised = result_raw + noise

        # Sum the output lines
        line_summed = torch.sum(result_noised, dim=-2)  # [...., 2*O]

        # Perform the subtraction of positive and negative lines
        shape = line_summed.shape
        axis_extended = torch.reshape(line_summed, shape[:-1] + (2, -1))  # [...., 2, O]
        intermediate = torch.moveaxis(axis_extended, -2, 0)  # [2, ...., O]
        result_noised_paired = intermediate[0] - intermediate[1]  # [...., O]

        return result_noised_paired


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
        # TODO: DAC Noise
        # * self.HARDWARE_VMAX -> convert to physical voltage
        input_voltage = int_quantized_input * self.hs.hardware_input_vmax
        return input_voltage

    def adc(self, tensor: torch.Tensor):
        # go from physical current to internal value space again
        adc_in = tensor * self.hs.hardware_output_current_scaling
        # print("adc_in")
        # print(torch.max(adc_in))
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


class MemristorLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_precision: int,
        converter_hardware_settings: DacAdcHardwareSettings,
    ):
        super().__init__()
        self.weight_precision = weight_precision
        self.memristors = torch.nn.ModuleList(
            [
                PairedMemristorArrayV2(in_features, out_features)
                for _ in range(weight_precision - 1)
            ]
        )
        self.converter = DacAdcPair(hardware_settings=converter_hardware_settings)
        self.input_factor = 1.0
        self.output_factor = 1.0

        self.initialized = False

    def init_from_linear_quant(
        self, activation_quant: ActivationQuantizer, linear_quant: LinearQuant
    ):
        quant_weights = linear_quant.weight_quantizer(linear_quant.weight).detach()
        # handle weight sign separately because integer division with negative numbers does not work as expected
        # for this case here, e.g. -5 // 2 = -3 instead of -2
        weights_sign = torch.sign(quant_weights)
        quant_weights_scaled_abs = torch.round(
            torch.absolute(quant_weights / linear_quant.weight_quantizer.scale)
        ).to(dtype=torch.int32)

        for i, bit in enumerate(reversed(range(0, self.weight_precision - 1))):
            # the weights we want to apply
            quant_weights_scaled_bit = quant_weights_scaled_abs // (2**bit)

            # the residual weights for the next step
            quant_weights_scaled_abs = quant_weights_scaled_abs % (2**bit)

            # re-apply sign and transpose
            quant_weights_scaled_transposed = torch.transpose(
                quant_weights_scaled_bit * weights_sign, 0, 1
            )  # [out, in] -> [in, out]

            # Arrays need flat input
            flat = torch.flatten(quant_weights_scaled_transposed).cpu()

            # positive numbers go in the positive line array, negative numbers as positive weight in the negative array
            positive_weights = torch.clamp(flat, 0, 1).numpy()
            negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()

            # apply negative voltage where a weight is set
            size = flat.shape[0]
            positive_cells = CellArrayCPU(size)
            negative_cells = CellArrayCPU(size)
            positive_cells.applyVoltage(positive_weights * -2.0)
            negative_cells.applyVoltage(negative_weights * -2.0)

            self.memristors[i].init_from_paired_cell_array_input_major(
                positive_cells, negative_cells
            )

        self.input_factor = 1.0 / (activation_quant.scale * activation_quant.quant_max)
        self.output_factor = (
            linear_quant.weight_quantizer.scale
            * activation_quant.scale
            * activation_quant.quant_max
        )
        self.initialized = True

    def forward(self, inputs: torch.Tensor):
        assert self.initialized
        inp = self.converter.dac(inputs * self.input_factor)
        mem_out = self.converter.adc(self.memristors[-1].forward(inp))
        for i, bit in enumerate(reversed(range(1, self.weight_precision - 1))):
            mem_out += self.converter.adc(self.memristors[i].forward(inp)) * (2**bit)
        out = mem_out * self.output_factor
        return out


def compute_correction_factor():

    from .synaptogen import CellArrayCPU, Iread

    correction_factors_paired = []
    correction_factors_single = []

    for i in range(1000):
        estimation_cells = CellArrayCPU(200)

        zero_offsets = []
        zero_offsets_nc = []
        for check in np.arange(0.001, 0.7, 0.1):
            out = Iread(estimation_cells, check)
            zero_offset = np.mean(out)
            zero_offsets_nc.append(zero_offset)
            zero_offsets.append(0)
            # print(f"check value: {check:.2} gives zero_offset {zero_offset}")

        zero_offset_nc = np.mean(zero_offsets_nc)

        # applyVoltage(estimation_cells, -2.5)
        estimation_cells.applyVoltage(np.asarray([-2.5] * 100 + [0.0] * 100))
        correction_factors = []
        correction_factors_nc = []
        for i, check in enumerate(np.arange(0.1, 0.7, 0.1)):
            cell_out = Iread(estimation_cells, check)
            out = cell_out[:100] - cell_out[100:]
            out_nc = cell_out[:100]
            correction_factor = check / np.mean(out - zero_offsets[i])
            correction_factor_nc = check / np.mean(out_nc - zero_offsets_nc[i])
            correction_factors.append(correction_factor)
            correction_factors_nc.append(correction_factor_nc)
            # print(f"check value: {check:.2} gives correction factor {correction_factor}")

        estimation_cells.applyVoltage(2.5)
        for i, check in enumerate(np.arange(0.1, 0.7, 0.1)):
            out = Iread(estimation_cells, check)
            # print(f"check value: {check:.2} gives min raw value: {np.min(out)}")

        correction_factor_paired = np.mean(correction_factors) / 0.6
        correction_factor_single = np.mean(correction_factors_nc) / 0.6

        correction_factors_paired.append(correction_factor_paired)
        correction_factors_single.append(correction_factor_single)

    print(f"correction factor paired: {np.mean(correction_factors_paired)}")
    print(f"correction factor single: {np.mean(correction_factors_single)}")
