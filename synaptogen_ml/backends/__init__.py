"""Pluggable device backends for the memristor crossbar modules.

A *backend* answers two questions for a crossbar module: which array module
evaluates the programmed cells in the forward pass, and how a list of pos/neg
weight vectors is turned into programmed cells. The default (``backend=None``
on every ``Memristor*`` module) is the exact historical Synaptogen code path;
``SynaptogenBackend`` is the same path behind the backend interface and exists
so the seam is tested. Measurement-based backends plug in here without
touching the Synaptogen code.
"""

from .base import CellBackend
from .synaptogen_backend import SynaptogenBackend

__all__ = ["CellBackend", "SynaptogenBackend"]
