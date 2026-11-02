"""Polynomial helpers for measured IV curves.

Coefficients are **ascending** (``c[0] + c[1] U + ...``), the convention of
``memristor_modules.util.poly_mul``; ``numpy.polyval`` and the Synaptogen
``LLRS``/``HHRS`` parameters use the reverse order (``to_polyval_order``).
"""

__all__ = [
    "fit_poly_zero_intercept",
    "eval_poly",
    "fit_quality",
    "to_polyval_order",
    "check_fit",
]

from typing import Optional, Tuple

import numpy as np


def _window_mask(U, window):
    mask = np.isfinite(U)
    if window is not None:
        lo, hi = window
        mask &= (U >= lo) & (U <= hi)
    return mask


def fit_poly_zero_intercept(
    U: np.ndarray,
    I: np.ndarray,
    degree: int,
    *,
    window: Optional[Tuple[float, float]] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Least-squares polynomial ``I(U) = sum_{p=1..degree} c_p U^p`` (no
    constant term, so ``I(0) = 0``). NaN samples are ignored.

    :param window: only fit samples with ``lo <= U <= hi``.
    :param weights: per-sample weights (e.g. ``1/|I|`` to equalise the relative
        error along a strongly nonlinear curve).
    :return: ascending coefficients ``float64[degree + 1]`` with ``c[0] = 0``.
    """
    U = np.asarray(U, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    mask = _window_mask(U, window) & np.isfinite(I)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        mask &= np.isfinite(weights)
    if mask.sum() < degree:
        raise ValueError(f"need at least {degree} finite samples, have {mask.sum()}")
    u, y = U[mask], I[mask]
    A = np.stack([u**p for p in range(1, degree + 1)], axis=1)
    if weights is not None:
        w = np.sqrt(weights[mask])
        A, y = A * w[:, None], y * w
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.concatenate([[0.0], coeffs])


def eval_poly(coeffs: np.ndarray, U) -> np.ndarray:
    """Evaluate ascending coefficients at ``U`` (Horner)."""
    U = np.asarray(U, dtype=np.float64)
    out = np.zeros_like(U)
    for c in np.asarray(coeffs, dtype=np.float64)[::-1]:
        out = out * U + c
    return out


def to_polyval_order(coeffs: np.ndarray) -> np.ndarray:
    """Ascending -> descending (``numpy.polyval`` / Synaptogen ``LLRS``)."""
    return np.asarray(coeffs, dtype=np.float64)[::-1].copy()


def fit_quality(
    coeffs: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    window: Optional[Tuple[float, float]] = None,
) -> float:
    """Relative RMS error ``||pred - I|| / ||I||`` over the finite samples in
    ``window``."""
    U = np.asarray(U, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    mask = _window_mask(U, window) & np.isfinite(I)
    pred = eval_poly(coeffs, U[mask])
    return float(
        np.sqrt(np.mean((pred - I[mask]) ** 2)) / np.sqrt(np.mean(I[mask] ** 2))
    )


def check_fit(coeffs: np.ndarray, u0: float, expected_sign: int) -> None:
    """Raise ``ValueError`` for non-finite coefficients or a read current at
    ``u0`` of the wrong sign (``expected_sign`` is +1 or -1)."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("polynomial fit has non-finite coefficients")
    value = float(eval_poly(coeffs, u0))
    if np.sign(value) != np.sign(expected_sign):
        raise ValueError(
            f"fitted current at u0={u0} is {value:.3e}, expected sign {expected_sign:+d}"
        )
