"""Credit-spread risk (CS01) and Z-spread, separated from rate risk.

Price = sum_i A_i * (1 + (z(t_i) + s)/f) ** (-n_i)
    z(t): risk-free zero curve      s: credit spread (Z-spread)

    DV01_curve : bump the whole curve z by 1bp, spread s held fixed
    CS01       : bump the spread s by 1bp, curve z held fixed

IMPORTANT (a finding worth writing in the README):
For a plain fixed-rate bullet bond and a *parallel* curve shift, z + s enters
the discount factor additively, so CS01 == DV01_curve exactly. The two risks
differ in (a) which *factor* moves (rates vs spread), (b) how much that factor
moves (spread vol >> rate vol for HY / in stress), and (c) non-parallel curve
moves, optionality, and default/recovery. So the useful comparison is
"sensitivity x typical factor move" (risk contribution), not the raw numbers.

Sign convention: positive number = loss for a +1bp move.
"""
from __future__ import annotations

from datetime import date

from scipy.optimize import brentq

from ..pricing.bond import Bond, ZeroCurve, dirty_price_from_curve
from ..pricing.curves import shift_curve

ONE_BP = 1e-4


def cs01(bond: Bond, settlement: date, curve: ZeroCurve, spread: float,
         notional: float = 100.0, bump: float = ONE_BP) -> float:
    """Central-difference loss for +1bp spread move, curve fixed."""
    up = dirty_price_from_curve(bond, settlement, curve, spread + bump)
    dn = dirty_price_from_curve(bond, settlement, curve, spread - bump)
    return (dn - up) / 2.0 * (notional / 100.0) * (ONE_BP / bump)


def dv01_curve(bond: Bond, settlement: date, curve: ZeroCurve, spread: float,
               notional: float = 100.0, bump: float = ONE_BP) -> float:
    """Central-difference loss for +1bp parallel curve move, spread fixed."""
    up = dirty_price_from_curve(bond, settlement, shift_curve(curve, bump), spread)
    dn = dirty_price_from_curve(bond, settlement, shift_curve(curve, -bump), spread)
    return (dn - up) / 2.0 * (notional / 100.0) * (ONE_BP / bump)


def spread_duration(bond: Bond, settlement: date, curve: ZeroCurve, spread: float,
                    bump: float = ONE_BP) -> float:
    """-(1/P) dP/ds, in years."""
    p0 = dirty_price_from_curve(bond, settlement, curve, spread)
    up = dirty_price_from_curve(bond, settlement, curve, spread + bump)
    dn = dirty_price_from_curve(bond, settlement, curve, spread - bump)
    return (dn - up) / (2.0 * p0 * bump)


def z_spread(bond: Bond, settlement: date, dirty_price: float, curve: ZeroCurve) -> float:
    """Constant spread over the zero curve that reprices the bond (Brent)."""
    g = lambda s: dirty_price_from_curve(bond, settlement, curve, s) - dirty_price
    lo, hi = -0.10, 3.0
    if g(lo) * g(hi) > 0:
        raise ValueError("dirty_price is outside the solvable range")
    return float(brentq(g, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200))
