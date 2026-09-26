"""Minimal zero-curve helpers (periodic compounding at the bond's frequency).

Nelson-Siegel / Svensson fitting is planned for v0.2; for now a flat curve and
a piecewise-linear curve are enough to test the spread/rate risk split.
"""
from __future__ import annotations

import numpy as np

from .bond import ZeroCurve


def flat_curve(rate: float) -> ZeroCurve:
    return lambda t: np.full_like(np.asarray(t, dtype=float), rate)


def linear_curve(tenors: list[float], rates: list[float]) -> ZeroCurve:
    """Linear interpolation, flat extrapolation outside the given tenors."""
    x = np.asarray(tenors, dtype=float)
    y = np.asarray(rates, dtype=float)
    return lambda t: np.interp(np.asarray(t, dtype=float), x, y)


def shift_curve(curve: ZeroCurve, bump: float) -> ZeroCurve:
    """Parallel shift (bump in decimal, 1bp = 1e-4)."""
    return lambda t: np.asarray(curve(t), dtype=float) + bump
