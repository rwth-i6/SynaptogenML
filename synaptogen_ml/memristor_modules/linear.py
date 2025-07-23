__all__ = ["MemristorLinear", "TiledMemristorLinear"]
import numpy as np
import torch
from torch import nn

from ..quant_modules import LinearQuant, ActivationQuantizer
from ..synaptogen import CellArrayCPU
from .memristor import DacAdcHardwareSettings, DacAdcPair, PairedMemristorArrayV2


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
        num_cycles_init: int,
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
                    for _ in range(num_cycles_init * 15):
                        positive_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
                        negative_cells.applyVoltage(np.random.uniform(-2.0, 2.0))

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
