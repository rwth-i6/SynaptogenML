"""Measurement-based ("empirical") device description.

``IVDataset`` holds measured current-voltage curves of many devices on a
common voltage grid, split by cycle and by programmed state. It is the data
the empirical backend samples cells from.
Adapters turn raw instrument exports into datasets
(``synaptogen_ml.empirical.adapters``), ``synthetic`` generates datasets from
the Synaptogen cell model for round-trip tests, ``fit`` holds the polynomial
helpers.

This package does not import torch, so it can be used from the torch-free
programming worker processes.
"""

from .dataset import IVDataset, from_hysteresis_loops, from_state_sweeps, segment_loop

__all__ = ["IVDataset", "from_hysteresis_loops", "from_state_sweeps", "segment_loop"]
