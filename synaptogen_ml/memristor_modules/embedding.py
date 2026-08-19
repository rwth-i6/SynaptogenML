__all__ = ["TiledMemristorEmbedding"]

import numpy as np
import torch
from torch import nn
from typing import Optional

from ..synaptogen import CellArrayCPU
from .memristor import DacAdcHardwareSettings, DacAdcPair, PairedMemristorArrayV2
from .config import CycleCorrectionSettings


class TiledMemristorEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_precision: int,
        converter_hardware_settings: DacAdcHardwareSettings,
        memristor_inputs: int,
        memristor_outputs: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight_precision = weight_precision

        self.input_tiling = (num_embeddings + memristor_inputs - 1) // memristor_inputs
        self.output_tiling = (
            embedding_dim + memristor_outputs - 1
        ) // memristor_outputs
        self.memristor_inputs = memristor_inputs
        self.memristor_outputs = memristor_outputs

        self.memristors = torch.nn.ModuleList(
            [
                PairedMemristorArrayV2(memristor_inputs, memristor_outputs)
                for _ in range(
                    (weight_precision - 1) * self.input_tiling * self.output_tiling
                )
            ]
        )
        self.converter = DacAdcPair(hardware_settings=converter_hardware_settings)
        self.input_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.output_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.initialized = False

    def get_memristor_index(self, bit_level_index, input_index, output_index):
        return (
            bit_level_index * (self.input_tiling * self.output_tiling)
            + input_index * self.output_tiling
            + output_index
        )

    def init_from_embedding_quant(
        self,
        embedding_quant,
        num_cycles_init: int,
        correction_settings: Optional[CycleCorrectionSettings],
    ):

        quant_weights = embedding_quant.weight_quantizer(
            embedding_quant.weight
        ).detach()
        weight_scale = embedding_quant.weight_quantizer.scale

        weights_sign = torch.sign(quant_weights)
        quant_weights_scaled_abs = torch.round(
            torch.absolute(quant_weights / weight_scale)
        ).to(dtype=torch.int32)

        fill_input = self.input_tiling * self.memristor_inputs - quant_weights.size(0)
        fill_output = self.output_tiling * self.memristor_outputs - quant_weights.size(
            1
        )

        for i, bit in enumerate(reversed(range(0, self.weight_precision - 1))):
            quant_weights_scaled_bit = quant_weights_scaled_abs // (2**bit)
            quant_weights_scaled_abs = quant_weights_scaled_abs % (2**bit)
            quant_weights_scaled = quant_weights_scaled_bit * weights_sign
            quant_weights_scaled_pad = nn.functional.pad(
                quant_weights_scaled,
                pad=(0, fill_output, 0, fill_input),
            )

            for j in range(self.input_tiling):
                for k in range(self.output_tiling):
                    slice_ = quant_weights_scaled_pad[
                        j * self.memristor_inputs : (j + 1) * self.memristor_inputs,
                        k * self.memristor_outputs : (k + 1) * self.memristor_outputs,
                    ]
                    flat = torch.flatten(slice_).cpu()
                    positive_weights = torch.clamp(flat, 0, 1).numpy()
                    negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()

                    size = flat.shape[0]
                    positive_cells = CellArrayCPU(size)
                    negative_cells = CellArrayCPU(size)
                    if (
                        correction_settings is not None
                        and correction_settings.ideal_programming
                    ):
                        positive_cells.r = (
                            np.ones_like(positive_weights) - positive_weights
                        )
                        negative_cells.r = (
                            np.ones_like(negative_weights) - negative_weights
                        )
                    else:
                        for _ in range(num_cycles_init * 15):
                            positive_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
                            negative_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
                        positive_cells.applyVoltage(2.0)
                        negative_cells.applyVoltage(2.0)
                        positive_cells.applyVoltage(positive_weights * -2.0)
                        negative_cells.applyVoltage(negative_weights * -2.0)

                        if correction_settings is not None:
                            for _ in range(correction_settings.num_cycles):
                                tensor = (
                                    np.ones_like(positive_weights)
                                    * correction_settings.test_input_value
                                )
                                pos = (
                                    positive_cells.I(tensor)
                                    * self.converter.hs.hardware_output_current_scaling
                                )
                                neg = (
                                    negative_cells.I(tensor)
                                    * self.converter.hs.hardware_output_current_scaling
                                )
                                pos_dev = np.abs(pos - positive_weights)
                                neg_dev = np.abs(neg - negative_weights)
                                pos_mask = (
                                    pos_dev > correction_settings.relative_deviation
                                )
                                neg_mask = (
                                    neg_dev > correction_settings.relative_deviation
                                )
                                positive_cells.applyVoltage(
                                    pos_mask * positive_weights * 2.0
                                )
                                positive_cells.applyVoltage(
                                    pos_mask * positive_weights * -2.0
                                )
                                positive_cells.applyVoltage(
                                    pos_mask * (1 - positive_weights) * -2.0
                                )
                                positive_cells.applyVoltage(
                                    pos_mask * (1 - positive_weights) * 2.0
                                )
                                negative_cells.applyVoltage(
                                    neg_mask * negative_weights * 2.0
                                )
                                negative_cells.applyVoltage(
                                    neg_mask * negative_weights * -2.0
                                )
                                negative_cells.applyVoltage(
                                    neg_mask * (1 - negative_weights) * -2.0
                                )
                                negative_cells.applyVoltage(
                                    neg_mask * (1 - negative_weights) * 2.0
                                )

                    index = self.get_memristor_index(i, j, k)
                    self.memristors[index].init_from_paired_cell_array_input_major(
                        positive_cells, negative_cells
                    )

        self.input_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.output_factor = torch.nn.Parameter(
            torch.as_tensor(weight_scale).detach().clone(),
            requires_grad=False,
        )
        self.initialized = True

    def forward(self, indices: torch.Tensor) -> torch.Tensor:

        assert self.initialized

        orig_shape = indices.shape
        flat_idx = indices.reshape(-1).to(dtype=torch.long)
        N = flat_idx.shape[0]
        out = torch.zeros(N, self.embedding_dim, device=indices.device)

        tile_of = flat_idx // self.memristor_inputs  # [N] which input tile
        pos_of = flat_idx % self.memristor_inputs  # [N] line within that tile

        for j in range(self.input_tiling):
            mask = tile_of == j
            if not torch.any(mask):
                continue
            rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
            G = rows.shape[0]
            pos = pos_of[rows]  # [G]

            local = torch.zeros(G, self.memristor_inputs, device=indices.device)
            local[torch.arange(G, device=indices.device), pos] = 1.0
            dac_in = self.converter.dac(local * self.input_factor)  # [G, mem_in]

            group_out = torch.zeros(G, self.embedding_dim, device=indices.device)
            for i, bit in enumerate(reversed(range(1, self.weight_precision))):
                start_index = self.get_memristor_index(i, j, 0)
                outputs = torch.concatenate(
                    [
                        self.converter.adc(
                            self.memristors[start_index + k].forward(dac_in)
                        )
                        for k in range(self.output_tiling)
                    ],
                    dim=-1,
                )  # [G, pad_out]
                outputs = outputs[..., : self.embedding_dim]
                group_out = group_out + outputs * (2 ** (bit - 1))

            out[rows] = group_out * self.output_factor

        if self.padding_idx is not None:
            out[flat_idx == self.padding_idx] = 0.0

        return out.reshape(*orig_shape, self.embedding_dim)
