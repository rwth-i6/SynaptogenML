__all__ = ["MemristorConv1d", "MemristorConv2d"]
from typing import Literal, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..quant_modules import ActivationQuantizer, Conv1DQuant, Conv2dQuant
from ..synaptogen import CellArrayCPU
from .memristor import DacAdcHardwareSettings, DacAdcPair, PairedMemristorArrayV2


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
        num_cycles_init: int,
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
            for _ in range(num_cycles_init * 15):
                positive_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
                negative_cells.applyVoltage(np.random.uniform(-2.0, 2.0))

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
        num_cycles_init: int,
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
            for _ in range(num_cycles_init * 15):
                positive_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
                negative_cells.applyVoltage(np.random.uniform(-2.0, 2.0))

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
