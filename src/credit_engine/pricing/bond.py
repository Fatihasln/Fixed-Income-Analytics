"""Fixed-rate bullet bond: cash flows, accrued interest, clean/dirty price.

Conventions (documented on purpose, because most 'bugs' in bond code are
convention mismatches):

* Prices are quoted per 100 of par. Position size is handled by a separate
  ``notional`` argument in the risk functions.
* Yields are annualised, periodically compounded at the coupon frequency
  ("street convention"): DF(n) = (1 + y/f) ** (-n), n = coupon periods.
* Settlement on a coupon date: that coupon is already paid (accrued = 0) and
  the next coupon is one full period away.
* Coupon dates are generated backwards from maturity (so a 31 Aug maturity
  gives 28/29 Feb and 31 Aug coupons - no drift).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
from dateutil.relativedelta import relativedelta

from .daycount import DayCount, period_fractions

ZeroCurve = Callable[[np.ndarray], np.ndarray]  # years -> zero rate (periodic comp.)


@dataclass(frozen=True)
class Cashflows:
    """Remaining cash flows seen from a settlement date (per 100 par)."""

    periods: np.ndarray   # time to each flow in coupon periods: w, w+1, w+2, ...
    amounts: np.ndarray   # coupon (and par at the end)
    accrued: float        # accrued interest per 100 par
    last_coupon: date
    next_coupon: date
    frequency: int

    @property
    def years(self) -> np.ndarray:
        return self.periods / self.frequency


@dataclass(frozen=True)
class Bond:
    coupon_rate: float                      # annual, decimal (0.05 = 5%)
    maturity: date
    frequency: int = 2                      # coupons per year
    day_count: DayCount = DayCount.THIRTY_360

    def __post_init__(self) -> None:
        if self.frequency not in (1, 2, 4, 12):
            raise ValueError("frequency must be 1, 2, 4 or 12")
        if self.coupon_rate < 0:
            raise ValueError("coupon_rate must be >= 0")

    # ------------------------------------------------------------------ dates
    def _coupon_dates(self, settlement: date) -> tuple[date, list[date]]:
        """Return (last coupon <= settlement, remaining coupon dates ascending)."""
        if settlement >= self.maturity:
            raise ValueError("settlement must be before maturity")
        step = 12 // self.frequency
        remaining: list[date] = []
        k = 0
        while True:
            d = self.maturity - relativedelta(months=step * k)
            if d <= settlement:
                last = d
                break
            remaining.append(d)
            k += 1
        remaining.reverse()
        return last, remaining

    # -------------------------------------------------------------- cashflows
    def cashflows(self, settlement: date) -> Cashflows:
        last, remaining = self._coupon_dates(settlement)
        nxt = remaining[0]
        elapsed, rem = period_fractions(
            self.day_count, last, settlement, nxt, self.frequency
        )
        coupon = 100.0 * self.coupon_rate / self.frequency
        n = len(remaining)
        amounts = np.full(n, coupon)
        amounts[-1] += 100.0
        periods = rem + np.arange(n, dtype=float)
        return Cashflows(
            periods=periods,
            amounts=amounts,
            accrued=coupon * elapsed,
            last_coupon=last,
            next_coupon=nxt,
            frequency=self.frequency,
        )

    def accrued_interest(self, settlement: date) -> float:
        return self.cashflows(settlement).accrued


# ------------------------------------------------------------------- pricing
def dirty_price(bond: Bond, settlement: date, ytm: float) -> float:
    """Full price (clean + accrued) per 100 par at a periodic-compounded yield."""
    cf = bond.cashflows(settlement)
    f = cf.frequency
    return float(np.sum(cf.amounts * (1.0 + ytm / f) ** (-cf.periods)))


def clean_price(bond: Bond, settlement: date, ytm: float) -> float:
    cf = bond.cashflows(settlement)
    f = cf.frequency
    dirty = float(np.sum(cf.amounts * (1.0 + ytm / f) ** (-cf.periods)))
    return dirty - cf.accrued


def dirty_price_from_curve(
    bond: Bond,
    settlement: date,
    zero_curve: ZeroCurve,
    spread: float = 0.0,
) -> float:
    """Discount each flow at zero(t) + spread (both periodic-compounded).

    This is what lets us separate two risk factors:
      * shifting ``zero_curve``  -> interest-rate risk (DV01)
      * shifting ``spread``      -> credit-spread risk (CS01)
    With a flat curve z = y - s this reproduces ``dirty_price(..., y)`` exactly.
    """
    cf = bond.cashflows(settlement)
    f = cf.frequency
    z = np.asarray(zero_curve(cf.years), dtype=float)
    return float(np.sum(cf.amounts * (1.0 + (z + spread) / f) ** (-cf.periods)))
