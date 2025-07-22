__all__ = [
    "poly_mul",
    "poly_mul_horner",
    "randn_broadcast",
    "compute_correction_factor",
]
from typing import Optional, Tuple

import numpy as np
import torch


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


def compute_correction_factor():
    from ..synaptogen import CellArrayCPU, Iread

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
