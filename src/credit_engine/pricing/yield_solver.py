"""Yield-to-maturity solvers: Brent (robust) and Newton (fast, analytic slope)."""
from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from .bond import Bond, dirty_price

PriceType = Literal["clean", "dirty"]

_Y_LO, _Y_HI = -0.5, 5.0   # search bracket for the yield (-50% .. 500%)


def _target_dirty(bond: Bond, settlement: date, price: float, price_type: PriceType) -> float:
    if price <= 0:
        raise ValueError("price must be positive")
    if price_type == "clean":
        return price + bond.accrued_interest(settlement)
    if price_type == "dirty":
        return price
    raise ValueError("price_type must be 'clean' or 'dirty'")


def ytm_brent(bond: Bond, settlement: date, price: float,
              price_type: PriceType = "clean", tol: float = 1e-14) -> float:
    """Yield via Brent's method. Price is strictly decreasing in y, so a sign
    change on the bracket guarantees a unique root."""
    target = _target_dirty(bond, settlement, price, price_type)
    g = lambda y: dirty_price(bond, settlement, y) - target
    lo, hi = _Y_LO, _Y_HI
    if g(lo) * g(hi) > 0:
        raise ValueError(f"price {price} is outside the solvable range for this bond")
    return float(brentq(g, lo, hi, xtol=tol, rtol=1e-14, maxiter=200))


def ytm_newton(bond: Bond, settlement: date, price: float,
               price_type: PriceType = "clean", y0: float | None = None,
               tol: float = 1e-12, max_iter: int = 50) -> float:
    """Yield via Newton-Raphson with the analytic dP/dy.

    Faster than Brent but can diverge for deeply distressed prices; keep Brent
    as the default and use this one as an independent cross-check.
    """
    target = _target_dirty(bond, settlement, price, price_type)
    cf = bond.cashflows(settlement)
    f = cf.frequency
    y = bond.coupon_rate if y0 is None else y0
    for _ in range(max_iter):
        base = 1.0 + y / f
        if base <= 0.05:
            raise RuntimeError("Newton left the valid domain; use ytm_brent")
        pv = np.sum(cf.amounts * base ** (-cf.periods))
        slope = np.sum(-cf.amounts * (cf.periods / f) * base ** (-cf.periods - 1.0))
        step = (pv - target) / slope
        y -= step
        if abs(step) < tol:
            return float(y)
    raise RuntimeError("Newton did not converge; use ytm_brent")


def solve_ytm(bond: Bond, settlement: date, price: float,
              price_type: PriceType = "clean", method: Literal["brent", "newton"] = "brent") -> float:
    if method == "brent":
        return ytm_brent(bond, settlement, price, price_type)
    if method == "newton":
        return ytm_newton(bond, settlement, price, price_type)
    raise ValueError("method must be 'brent' or 'newton'")
