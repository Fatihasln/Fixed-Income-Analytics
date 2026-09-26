"""Interest-rate risk measures for a fixed-rate bond.

Notation (per 100 par, yield y compounded f times a year, n_i = periods to flow i):

    P      = sum_i  A_i (1+y/f)^(-n_i)                       dirty price
    D_mac  = sum_i (n_i/f) * PV_i / P                        Macaulay duration (years)
    D_mod  = D_mac / (1 + y/f)                               = -(1/P) dP/dy
    C      = (1/P) sum_i A_i n_i (n_i+1) / f^2 (1+y/f)^(-n_i-2)   = (1/P) d2P/dy2
    DV01   = D_mod * P * 1e-4                                price change for +1bp

Everything is measured on the *dirty* price (that is the actual money at risk).
Sign convention: DV01 is reported as a positive number = loss for a +1bp move.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..pricing.bond import Bond, dirty_price

ONE_BP = 1e-4


def _pv_terms(bond: Bond, settlement: date, ytm: float):
    cf = bond.cashflows(settlement)
    f = cf.frequency
    base = 1.0 + ytm / f
    pv = cf.amounts * base ** (-cf.periods)
    return cf, base, pv


# ------------------------------------------------------------------ analytic
def macaulay_duration(bond: Bond, settlement: date, ytm: float) -> float:
    cf, _, pv = _pv_terms(bond, settlement, ytm)
    return float(np.sum(cf.years * pv) / np.sum(pv))


def modified_duration(bond: Bond, settlement: date, ytm: float) -> float:
    f = bond.frequency
    return macaulay_duration(bond, settlement, ytm) / (1.0 + ytm / f)


def convexity(bond: Bond, settlement: date, ytm: float) -> float:
    cf, base, pv = _pv_terms(bond, settlement, ytm)
    f = cf.frequency
    n = cf.periods
    second = np.sum(cf.amounts * n * (n + 1.0) / f**2 * base ** (-n - 2.0))
    return float(second / np.sum(pv))


def dv01(bond: Bond, settlement: date, ytm: float, notional: float = 100.0) -> float:
    """Money lost for a +1bp yield move on a position of ``notional`` par."""
    price = dirty_price(bond, settlement, ytm)
    position_value = price / 100.0 * notional
    return modified_duration(bond, settlement, ytm) * position_value * ONE_BP


# ------------------------------------------------------------ bump-and-reprice
def effective_duration(bond: Bond, settlement: date, ytm: float, bump: float = ONE_BP) -> float:
    """(P- - P+) / (2 P0 dy): finite-difference duration, no formula needed."""
    p0 = dirty_price(bond, settlement, ytm)
    up = dirty_price(bond, settlement, ytm + bump)
    dn = dirty_price(bond, settlement, ytm - bump)
    return (dn - up) / (2.0 * p0 * bump)


def effective_convexity(bond: Bond, settlement: date, ytm: float, bump: float = ONE_BP) -> float:
    """(P+ + P- - 2 P0) / (P0 dy^2)."""
    p0 = dirty_price(bond, settlement, ytm)
    up = dirty_price(bond, settlement, ytm + bump)
    dn = dirty_price(bond, settlement, ytm - bump)
    return (up + dn - 2.0 * p0) / (p0 * bump**2)


def price_change_estimate(mod_dur: float, conv: float, price: float, dy: float) -> float:
    """Second-order Taylor: dP ~ P * (-D_mod dy + 0.5 C dy^2)."""
    return price * (-mod_dur * dy + 0.5 * conv * dy**2)


# ------------------------------------------------------------------- summary
@dataclass(frozen=True)
class RateRisk:
    ytm: float
    clean_price: float
    dirty_price: float
    accrued: float
    macaulay: float
    modified: float
    convexity: float
    dv01: float
    effective_duration: float
    effective_convexity: float


def rate_risk_report(bond: Bond, settlement: date, ytm: float, notional: float = 100.0) -> RateRisk:
    dirty = dirty_price(bond, settlement, ytm)
    accrued = bond.accrued_interest(settlement)
    return RateRisk(
        ytm=ytm,
        clean_price=dirty - accrued,
        dirty_price=dirty,
        accrued=accrued,
        macaulay=macaulay_duration(bond, settlement, ytm),
        modified=modified_duration(bond, settlement, ytm),
        convexity=convexity(bond, settlement, ytm),
        dv01=dv01(bond, settlement, ytm, notional),
        effective_duration=effective_duration(bond, settlement, ytm),
        effective_convexity=effective_convexity(bond, settlement, ytm),
    )
