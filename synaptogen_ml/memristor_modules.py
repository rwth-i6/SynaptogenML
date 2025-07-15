from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .quant_modules import Conv1DQuant, LinearQuant, ActivationQuantizer, Conv2dQuant
from .synaptogen import CellArrayCPU


@torch.compile
def poly_mul(coefficients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a polynomial function on all input elements.
    :param coefficients: polynomial coefficients of shape [P], in ascending order of degree
    :param inputs: inputs ("x") to be evaluated in the shape of [..., I]
    :return: [...., I] outputs
    """
    exponents = torch.arange(0, coefficients.shape[-1], device=inputs.device)  # P
    inputs = inputs.unsqueeze(-1)  # [..., I, 1]
    result = coefficients * (inputs**exponents)  # [..., I, P]
    return result.sum(dim=-1)  # [..., I]


@torch.compile
def poly_mul_horner(coefficients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a polynomial function on all input elements using Horner's optimized method.

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
    Compute gaussian noise for all dims except the leading ``num_broadcast_dims`` and
    broadcast along these.

    Saves memory and computation time for large batch sizes.

    :param shape: Shape of the tensor to be created, split in [...num_broadcast_dims, ..dims].
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
        result_raw = (
            result_low * (1 - self.r) + result_high * self.r
        )  # [..., ...A, I, O]
        return result_raw

    def compute_noise(
        self, result_raw: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """

        :param result_raw: [..., ...A, I, O]
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
        """
        :param inputs: [...B, I]
        :return: [...B, ...A, O]
        """

        result_raw = self.compute_raw_output(inputs)
        noise = self.compute_noise(result_raw, inputs)
        result_noised = result_raw + noise

        return torch.sum(
            result_noised, dim=-2
        )  # [...B, ...A, I, O] -> sum reduce I -> [...B, ...A, O]


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
        self.linear_bias = None

    def init_from_linear_quant(
        self, activation_quant: ActivationQuantizer, linear_quant: LinearQuant
    ):
        quant_weights = linear_quant.weight_quantizer(linear_quant.weight).detach()
        self.linear_bias = linear_quant.bias
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
        if self.linear_bias is not None:
            out = out + self.linear_bias
        return out


class TiledMemristorLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_precision: int,
        converter_hardware_settings: DacAdcHardwareSettings,
        memristor_inputs: int,
        memristor_outputs: int,
    ):
        super().__init__()
        self.weight_precision = weight_precision
        self.input_tiling = (in_features + memristor_inputs - 1) // memristor_inputs
        self.output_tiling = (out_features + memristor_outputs - 1) // memristor_outputs
        self.memristor_inputs = memristor_inputs
        self.memristor_outputs = memristor_outputs

        self.in_features = in_features
        self.out_features = out_features

        self.memristors = torch.nn.ModuleList(
            [
                PairedMemristorArrayV2(memristor_inputs, memristor_outputs)
                for _ in range(
                    (weight_precision - 1) * self.input_tiling * self.output_tiling
                )
            ]
        )
        self.converter = DacAdcPair(hardware_settings=converter_hardware_settings)
        self.input_factor = 1.0
        self.output_factor = 1.0

        self.initialized = False
        self.bias = None

    def get_memristor_index(self, bit_level_index, input_index, output_index):
        return (
            bit_level_index * (self.input_tiling * self.output_tiling)
            + input_index * self.output_tiling
            + output_index
        )

    def init_from_linear_quant(
        self,
        activation_quant: ActivationQuantizer,
        linear_quant: LinearQuant,
        num_cycles: int,
    ):
        quant_weights = linear_quant.weight_quantizer(linear_quant.weight).detach()
        self.bias = linear_quant.bias
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

            fill_input = (
                self.input_tiling * self.memristor_inputs
                - quant_weights_scaled_transposed.size(0)
            )
            fill_output = (
                self.output_tiling * self.memristor_outputs
                - quant_weights_scaled_transposed.size(1)
            )
            quant_weights_scaled_transposed_pad = nn.functional.pad(
                input=quant_weights_scaled_transposed,
                pad=(0, fill_output, 0, fill_input),
            )

            for j in range(self.input_tiling):
                for k in range(self.output_tiling):
                    quant_weights_scaled_transposed_slice = (
                        quant_weights_scaled_transposed_pad[
                            j * self.memristor_inputs : (j + 1) * self.memristor_inputs,
                            k * self.memristor_outputs : (k + 1)
                            * self.memristor_outputs,
                        ]
                    )

                    # Arrays need flat input
                    flat = torch.flatten(quant_weights_scaled_transposed_slice).cpu()

                    # positive numbers go in the positive line array, negative numbers as positive weight in the negative array
                    positive_weights = torch.clamp(flat, 0, 1).numpy()
                    negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()

                    # apply negative voltage where a weight is set
                    size = flat.shape[0]
                    positive_cells = CellArrayCPU(size)
                    negative_cells = CellArrayCPU(size)
                    for _ in range(num_cycles * 50):
                        positive_cells.applyVoltage(numpy.random.uniform(-2.0, 2.0))
                        negative_cells.applyVoltage(numpy.random.uniform(-2.0, 2.0))

                    positive_cells.applyVoltage(2.0)
                    negative_cells.applyVoltage(2.0)
                    positive_cells.applyVoltage(positive_weights * -2.0)
                    negative_cells.applyVoltage(negative_weights * -2.0)

                    index = self.get_memristor_index(i, j, k)

                    self.memristors[index].init_from_paired_cell_array_input_major(
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
        fill_input = self.input_tiling * self.memristor_inputs - inp.size(-1)
        inp = nn.functional.pad(inp, (0, fill_input))
        mem_out = torch.zeros(
            inputs.shape[:-1] + (self.out_features,), device=inp.device
        )
        for i, bit in enumerate(reversed(range(1, self.weight_precision))):
            inputs = []
            for j in range(self.input_tiling):
                input_slice = inp[
                    ..., j * self.memristor_inputs : (j + 1) * self.memristor_inputs
                ]
                start_index = self.get_memristor_index(i, j, 0)
                outputs = torch.concatenate(
                    [
                        self.converter.adc(
                            self.memristors[start_index + k].forward(input_slice)
                        )
                        for k in range(self.output_tiling)
                    ],
                    dim=-1,
                )  # [mem_in, pad_out]

                # discard the padding in the last axis
                outputs = outputs[..., : self.out_features]
                inputs.append(outputs)
            mem_sum = torch.sum(torch.stack(inputs, dim=0), dim=0)
            mem_out += mem_sum * (2 ** (bit - 1))
        out = mem_out * self.output_factor
        if self.bias is not None:
            out = out + self.bias
        return out


class MemristorConv1d(nn.Module):
    """
    Memristive 1d-convolution as required for the implementation of a Conformer block.
    Currently, supports only groups==in_dim==out_dim
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[int, Literal["same", "valid"]] = 0,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        groups: int,
        weight_precision: int,
        converter_hardware_settings: DacAdcHardwareSettings,
    ):
        super().__init__()

        assert in_channels > 0
        self.in_channels = in_channels
        assert out_channels > 0
        self.out_channels = out_channels
        assert groups > 0
        assert in_channels % groups == 0
        self.groups = groups
        if not groups == in_channels == out_channels:
            raise NotImplementedError(
                "Conv1d currently only supports groups == in_channels == out_channels"
            )
        assert kernel_size > 0
        self.kernel_size = kernel_size
        if isinstance(padding, int):
            assert padding >= 0
        else:
            assert padding in ["same", "valid"]
        self.padding = padding
        assert padding_mode in ["zeros", "reflect", "replicate", "circular"]
        self.padding_mode = padding_mode
        assert stride > 0
        self.stride = stride

        self.memristors = torch.nn.ModuleList(
            [
                PairedMemristorArrayV2(
                    in_features=kernel_size,
                    out_features=1,
                    additional_axes=(
                        groups,
                        out_channels // groups,
                        in_channels // groups,
                    ),
                )
                for _ in range(weight_precision - 1)
            ]
        )

        self.converter = DacAdcPair(hardware_settings=converter_hardware_settings)
        assert weight_precision > 0
        self.weight_precision = weight_precision

        self.input_factor = 1.0
        self.output_factor = 1.0

        self.initialized = False
        self.bias = None

    def init_from_conv_quant(
        self,
        activation_quant: ActivationQuantizer,
        conv_quant: Conv1DQuant,
        num_cycles: int,
    ):
        quant_weights = conv_quant.weight_quantizer(conv_quant.weight).detach()
        self.bias = conv_quant.bias

        # handle weight sign separately because integer division with negative numbers does not work as expected
        # for this case here, e.g. -5 // 2 = -3 instead of -2
        weights_sign = torch.sign(quant_weights)
        quant_weights_scaled_abs = torch.round(
            torch.absolute(quant_weights / conv_quant.weight_quantizer.scale)
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
            for _ in range(num_cycles * 50):
                positive_cells.applyVoltage(numpy.random.uniform(-2.0, 2.0))
                negative_cells.applyVoltage(numpy.random.uniform(-2.0, 2.0))

            positive_cells.applyVoltage(2.0)
            negative_cells.applyVoltage(2.0)
            positive_cells.applyVoltage(positive_weights * -2.0)
            negative_cells.applyVoltage(negative_weights * -2.0)

            # tensor = np.ones_like(positive_weights) * 0.6
            # pos = positive_cells.I(tensor) * self.converter.hs.hardware_output_current_scaling
            # neg = negative_cells.I(tensor) * self.converter.hs.hardware_output_current_scaling
            # mix = pos - neg
            # mix_dev = mix - (positive_weights - negative_weights)
            # mix_mask = mix_dev > 1
            # pos_dev = pos - positive_weights
            # neg_dev = neg - negative_weights

            self.memristors[i].init_from_paired_cell_array_input_major(
                positive_cells, negative_cells
            )

        self.input_factor = 1.0 / (activation_quant.scale * activation_quant.quant_max)
        self.output_factor = (
            conv_quant.weight_quantizer.scale
            * activation_quant.scale
            * activation_quant.quant_max
        )
        self.initialized = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies 1d-convolution.

        :param inputs: [..., F, T]
        :return: [..., F', T']
        """

        assert self.initialized

        in_ndim = inputs.ndim
        inputs = self.converter.dac(inputs * self.input_factor)
        inputs = inputs.transpose(-2, -1)  # [..., T, F]
        if isinstance(self.padding, int):
            padding_amount = self.padding
        elif self.padding == "same":
            padding_amount = self.kernel_size // 2
        elif self.padding == "valid":
            padding_amount = 0
        else:
            raise ValueError(f"Unknown padding mode: {self.padding}")
        if padding_amount > 0:
            mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            inputs = F.pad(inputs, (0, 0, padding_amount, padding_amount), mode=mode)

        in0 = inputs
        in1 = in0.unfold(-2, self.kernel_size, self.stride)  # [..., T', F, C]
        in2 = in1.view(
            *in1.shape[:-2], self.groups, -1, self.kernel_size
        )  # [..., T', G, F//G, C]
        in3 = in2.unsqueeze(-2)  # [..., T', G, F//G, 1, C]
        in4 = in3.expand(
            *([-1] * (in3.ndim - 2)), self.out_channels // self.groups, -1
        )  # [..., T', G, F//G, O//G, C]

        mem_out = self.converter.adc(
            self.memristors[-1].forward(in4)
        )  # [..., T', G, F//G, O//G, 1]
        for i, bit in enumerate(reversed(range(1, self.weight_precision - 1))):
            mem_out += self.converter.adc(self.memristors[i].forward(in4)) * (
                2 ** (bit)
            )
        mem_out *= self.output_factor

        result = mem_out.reshape(*mem_out.shape[: in_ndim - 1], -1)  # [..., T', O]
        if self.bias is not None:
            result = result + self.bias
        return result.transpose(-2, -1)  # [..., O, T']


class MemristorConv2d(nn.Module):
    """
    Memristive 2d-convolution
    Currently, supports groups==1 and groups==in_dim==out_dim

    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], Literal["same", "valid"]] = 0,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        groups: int,
        weight_precision: int,
        converter_hardware_settings: DacAdcHardwareSettings,
    ):
        super().__init__()

        assert in_channels > 0
        self.in_channels = in_channels
        assert out_channels > 0
        self.out_channels = out_channels
        assert groups > 0
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.groups = groups
        if not (groups == in_channels == out_channels or groups == 1):
            raise NotImplementedError(
                "Conv2d currently only supports groups == in_channels == out_channels or groups == 1"
            )
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert all(x > 0 for x in kernel_size)
        self.kernel_size = kernel_size
        if isinstance(padding, int):
            padding = (padding, padding)

        if isinstance(padding, tuple):
            assert all(x >= 0 for x in padding)
        else:
            assert padding in ["same", "valid"]
        self.padding = padding
        assert padding_mode in ["zeros", "reflect", "replicate", "circular"]
        self.padding_mode = padding_mode

        if isinstance(stride, int):
            stride = (stride, stride)
        assert all(x > 0 for x in stride)
        self.stride = stride


        self.memristors = torch.nn.ModuleList(
            [
                PairedMemristorArrayV2(
                    in_features=kernel_size[1],
                    out_features=1,
                    additional_axes=(
                        groups,
                        out_channels // groups,
                        in_channels // groups,
                        kernel_size[0],
                    ),
                )
                for _ in range(weight_precision - 1)
            ]
        )

        self.converter = DacAdcPair(hardware_settings=converter_hardware_settings)
        assert weight_precision > 0
        self.weight_precision = weight_precision

        self.input_factor = 1.0
        self.output_factor = 1.0

        self.initialized = False
        self.bias = None

    def init_from_conv_quant(
        self,
        activation_quant: ActivationQuantizer,
        conv_quant: Conv2dQuant,
        num_cycles: int,
    ):
        quant_weights = conv_quant.weight_quantizer(conv_quant.weight).detach()
        self.bias = conv_quant.bias

        # handle weight sign separately because integer division with negative numbers does not work as expected
        # for this case here, e.g. -5 // 2 = -3 instead of -2
        weights_sign = torch.sign(quant_weights)
        quant_weights_scaled_abs = torch.round(
            torch.absolute(quant_weights / conv_quant.weight_quantizer.scale)
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
            for _ in range(num_cycles * 50):
                positive_cells.applyVoltage(numpy.random.uniform(-2.0, 2.0))
                negative_cells.applyVoltage(numpy.random.uniform(-2.0, 2.0))

            positive_cells.applyVoltage(2.0)
            negative_cells.applyVoltage(2.0)
            positive_cells.applyVoltage(positive_weights * -2.0)
            negative_cells.applyVoltage(negative_weights * -2.0)

            self.memristors[i].init_from_paired_cell_array_input_major(
                positive_cells, negative_cells
            )

        self.input_factor = 1.0 / (activation_quant.scale * activation_quant.quant_max)
        self.output_factor = (
            conv_quant.weight_quantizer.scale
            * activation_quant.scale
            * activation_quant.quant_max
        )
        self.initialized = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies 2d-convolution.

        :param inputs: [..., F, T]
        :return: [..., F', T']
        """
        assert self.initialized

        inputs = self.converter.dac(inputs * self.input_factor)
        inputs = inputs.transpose(-2, -1)  # [..., T, F]
        batch_size, in_channels, time_dim, feature_dim = inputs.shape

        if isinstance(self.padding, tuple):
            padding_amount = self.padding
        elif self.padding == "same":
            padding_amount = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        elif self.padding == "valid":
            padding_amount = (0, 0)
        else:
            raise ValueError(f"Unknown padding mode: {self.padding}")
        if any(x > 0 for x in padding_amount):
            mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            inputs = F.pad(
                inputs,
                (
                    padding_amount[0],
                    padding_amount[0],
                    padding_amount[1],
                    padding_amount[1],
                ),
                mode=mode,
            )

        in0 = inputs
        in1 = in0.unfold(-2, self.kernel_size[0], self.stride[0]).unfold(
            -2, self.kernel_size[1], self.stride[1]
        )  # [B, in_c, T//S[0], F//S[1], K_[0], K[1]]
        in2 = in1.reshape(
            batch_size, in_channels, -1, self.kernel_size[0], self.kernel_size[1]
        )  # [Batch, in_channels, T//S[0] * F//S[1], K_[0], K_[1]]
        in3 = in2.reshape(
            batch_size,
            self.groups,
            -1,
            in2.shape[2],
            self.kernel_size[0],
            self.kernel_size[1],
        )  # [Batch, groups, in_channels // groups, T//S[0] * F//S[1], K_[0], K_[1]]
        in4 = in3.permute(0, 3, 1, 2, 4, 5).unsqueeze(
            3
        )  # [Batch, T//S[0] * F//S[1], groups, 1, in_channels//groups, K_[0], K_[1]]
        in5 = in4.expand(
            batch_size,
            -1,
            self.groups,
            self.out_channels // self.groups,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )  # [Batch, T//S[0] * F//S[1], groups, in_channels//groups, out_channels//groups, K_[0], K_[1]]
        out = self.memristors[-1].forward(in5).sum([-3, -2, -1])
        out = out.reshape(batch_size, -1, self.out_channels)
        mem_out = self.converter.adc(
            out
        )  # [Batch, T//S[0] * F//S[1], out_channels//groups]
        for i, bit in enumerate(reversed(range(0, self.weight_precision - 1))):
            out = (
                self.memristors[i]
                .forward(in4)
                .sum([-3, -2, -1])
                .reshape(batch_size, -1, self.out_channels)
            )
            mem_out += self.converter.adc(out) * (2 ** (bit))

        mem_out *= self.output_factor
        result = mem_out.reshape(
            batch_size, in1.shape[2], in1.shape[3], self.out_channels
        )  # [Batch, out_channels, T//S[0], F//S[1]]
        if self.bias is not None:
            result = result + self.bias
        return result.permute(0, 3, 1, 2)  # [..., O, T']


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
