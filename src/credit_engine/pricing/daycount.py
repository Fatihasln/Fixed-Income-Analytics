"""Day-count conventions.

Only what a fixed-rate bullet bond needs: for a given coupon period we want
the fraction of the period that has *elapsed* (-> accrued interest) and the
fraction that *remains* (-> first exponent in the discounting formula).

Supported:
    30/360 US (bond basis)  - US corporate bonds
    ACT/ACT ICMA            - US Treasuries, Eurobonds
"""
from __future__ import annotations

import calendar
from datetime import date
from enum import Enum


class DayCount(str, Enum):
    THIRTY_360 = "30/360"
    ACT_ACT_ICMA = "ACT/ACT"


def _is_last_day_of_feb(d: date) -> bool:
    return d.month == 2 and d.day == calendar.monthrange(d.year, 2)[1]


def days_30_360_us(d1: date, d2: date) -> int:
    """30/360 US day count (ISDA/SIA 'bond basis' with the February rules).

    Rules, applied in this order:
      1. if both d1 and d2 are the last day of February -> D2 = 30
      2. if d1 is the last day of February              -> D1 = 30
      3. if D2 == 31 and D1 >= 30                       -> D2 = 30
      4. if D1 == 31                                    -> D1 = 30
    """
    dd1, dd2 = d1.day, d2.day
    if _is_last_day_of_feb(d1) and _is_last_day_of_feb(d2):
        dd2 = 30
    if _is_last_day_of_feb(d1):
        dd1 = 30
    if dd2 == 31 and dd1 >= 30:
        dd2 = 30
    if dd1 == 31:
        dd1 = 30
    return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (dd2 - dd1)


def period_fractions(
    convention: DayCount,
    last_coupon: date,
    settlement: date,
    next_coupon: date,
    frequency: int,
) -> tuple[float, float]:
    """Return (elapsed, remaining) fractions of the current coupon period.

    elapsed   = share of the period between last coupon and settlement
    remaining = 1 - elapsed

    Why ``remaining = 1 - elapsed`` and not a second day count from settlement
    to the next coupon: with 30/360 the two direct counts can add up to 361 or
    359 days when a coupon date is the 31st or the end of February. The industry
    standard (SIA: DSC = E - A) and QuantLib both define the days to the next
    coupon as the period length minus the accrued days, so the two pieces always
    sum to exactly one period.
    """
    if convention is DayCount.THIRTY_360:
        elapsed = days_30_360_us(last_coupon, settlement) / (360.0 / frequency)
    elif convention is DayCount.ACT_ACT_ICMA:
        elapsed = (settlement - last_coupon).days / (next_coupon - last_coupon).days
    else:  # pragma: no cover
        raise ValueError(f"Unsupported day count: {convention}")
    return elapsed, 1.0 - elapsed
