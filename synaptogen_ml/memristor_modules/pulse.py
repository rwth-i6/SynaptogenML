"""Pulse-count readout: one conductance per cell, linear by construction.

With *pulse-count input encoding* an input value is applied as a number of
identical read pulses at a fixed read voltage; the array output is the
accumulated charge, i.e. ``sum_i n_i * I_i`` with ``I_i`` the current of cell
``i`` at the read voltage. This is the recommended encoding for strongly
nonlinear / rectifying devices: the device IV shape never enters, only the
read current at one voltage.

The module keeps the array interface of ``MemristorArray`` (``forward([...B,
I]) -> [...B, ...A, O]``, same broadcasting of leading input dims against the
``additional_axes``), so the crossbar modules use it unchanged. Its input is
the DAC output voltage; ``n = inputs / pulse_voltage`` is the (signed, in
general fractional) pulse count. A DAC window of ``hardware_input_vmax =
dac_max * pulse_voltage`` makes every DAC step one pulse
(``EmpiricalBackend.suggested_hardware_settings``).

Readout noise: Johnson (``4 kT G BW``) and shot (``2 e I BW``) noise per pulse
with the physical constants, accumulated over ``|n|`` pulses (independent
pulses -> variances add) and summed over the inputs, so one Gaussian draw per
output is exact. ``bandwidth`` is a module parameter (at GΩ devices the
integration time matters). The process-wide readout-noise switch is
honoured for ``"off"``; ``"legacy"`` and ``"physical"`` both mean "on" here,
the legacy constants (``e = exp(1)``) are not used. The linear-IV oracle and
fast-inference switches are irrelevant: the forward is a matmul already.
"""

__all__ = ["PulseCountMemristorArray", "PairedPulseCountMemristorArray"]

from typing import Sequence

import numpy as np
import torch
from torch import nn

from .. import has_readout_noise

K_BOLTZMANN_T300 = 1.380649e-23 * 300
ELECTRON_CHARGE = 1.602176634e-19


class PulseCountMemristorArray(nn.Module):
    """
    :param in_features: input lines (I)
    :param out_features: output lines (O)
    :param additional_axes: extra leading axes of the cell grid (conv layouts)
    :param pulse_voltage: magnitude of the read pulse in volt (``|V_read|``);
        ``inputs / pulse_voltage`` is the pulse count.
    :param bandwidth: readout bandwidth in Hz for the noise model.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        additional_axes: Sequence[int] = (),
        pulse_voltage: float,
        bandwidth: float = 1e8,
    ):
        super().__init__()
        if pulse_voltage <= 0:
            raise ValueError("pulse_voltage must be positive (a magnitude)")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.num_additional_axes = len(additional_axes)
        # per-cell current per pulse (ampere), positive in read polarity
        self.I_pulse = nn.Parameter(
            torch.empty(tuple(additional_axes) + (in_features, out_features)),
            requires_grad=False,
        )
        self.pulse_voltage = float(pulse_voltage)
        self.bandwidth = float(bandwidth)
        self.kBT = K_BOLTZMANN_T300
        self.e = ELECTRON_CHARGE

    def init_from_cell_array_input_major(self, cells) -> None:
        """``cells.I_pulse``: flat ``[M]`` current per pulse, input-major."""
        values = np.asarray(cells.I_pulse, dtype=np.float32)
        if values.size != self.I_pulse.numel():
            raise ValueError(
                f"got {values.size} cells for an array of {self.I_pulse.numel()}"
            )
        self.I_pulse.data = torch.as_tensor(values).reshape(self.I_pulse.shape)

    def pulse_noise_variance(self) -> torch.Tensor:
        """``[...A, I, O]`` noise variance of a single read pulse per cell."""
        abs_i = torch.abs(self.I_pulse)
        johnson = 4 * self.kBT * (abs_i / self.pulse_voltage)
        shot = 2 * self.e * abs_i
        return self.bandwidth * (johnson + shot)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: ``[...B, I]`` DAC output voltages
        :return: ``[...B, ...A, O]`` accumulated currents
        """
        counts = inputs / self.pulse_voltage
        mean = torch.matmul(counts.unsqueeze(-2), self.I_pulse).squeeze(-2)
        if not has_readout_noise():
            return mean
        variance = torch.matmul(
            torch.abs(counts).unsqueeze(-2), self.pulse_noise_variance()
        ).squeeze(-2)
        return mean + torch.randn_like(mean) * torch.sqrt(variance)


class PairedPulseCountMemristorArray(nn.Module):
    """Differential pair ``pos - neg`` of ``PulseCountMemristorArray``."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pos = PulseCountMemristorArray(*args, **kwargs)
        self.neg = PulseCountMemristorArray(*args, **kwargs)

    def init_from_paired_cell_array_input_major(self, positive_cells, negative_cells):
        self.pos.init_from_cell_array_input_major(positive_cells)
        self.neg.init_from_cell_array_input_major(negative_cells)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pos.forward(inputs) - self.neg.forward(inputs)
