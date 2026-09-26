from datetime import date

import pytest

from credit_engine.pricing.bond import Bond, clean_price, dirty_price
from credit_engine.pricing.daycount import DayCount, days_30_360_us


def test_par_bond_prices_at_100_on_coupon_date():
    b = Bond(0.06, date(2030, 6, 15), frequency=2)
    s = date(2025, 6, 15)
    assert dirty_price(b, s, 0.06) == pytest.approx(100.0, abs=1e-10)
    assert b.accrued_interest(s) == 0.0


def test_accrued_half_period_30_360():
    # coupon dates 15 Jun / 15 Dec; settle 15 Sep = 90 days of 180 -> half a coupon
    b = Bond(0.06, date(2030, 12, 15), frequency=2)
    assert b.accrued_interest(date(2025, 9, 15)) == pytest.approx(1.5)


def test_clean_plus_accrued_is_dirty():
    b = Bond(0.05, date(2032, 3, 1), frequency=2)
    s = date(2026, 5, 20)
    assert clean_price(b, s, 0.07) + b.accrued_interest(s) == pytest.approx(dirty_price(b, s, 0.07))


def test_price_decreasing_in_yield_and_premium_discount():
    b = Bond(0.05, date(2032, 3, 1))
    s = date(2026, 5, 20)
    assert clean_price(b, s, 0.03) > clean_price(b, s, 0.05) > clean_price(b, s, 0.08)
    assert clean_price(b, s, 0.03) > 100 > clean_price(b, s, 0.08)


def test_zero_coupon_price():
    b = Bond(0.0, date(2030, 6, 15), frequency=2)
    s = date(2025, 6, 15)
    assert dirty_price(b, s, 0.04) == pytest.approx(100 / 1.02**10)


def test_settlement_on_or_after_maturity_rejected():
    b = Bond(0.05, date(2030, 6, 15))
    with pytest.raises(ValueError):
        b.cashflows(date(2030, 6, 15))


def test_schedule_end_of_month_does_not_drift():
    b = Bond(0.05, date(2030, 8, 31), frequency=2)
    last, remaining = b._coupon_dates(date(2026, 3, 10))
    assert last == date(2026, 2, 28)
    assert remaining[0] == date(2026, 8, 31)


def test_30_360_basic_rules():
    assert days_30_360_us(date(2025, 1, 31), date(2025, 3, 31)) == 60
    assert days_30_360_us(date(2025, 1, 15), date(2025, 7, 15)) == 180


def test_act_act_accrued():
    b = Bond(0.04, date(2030, 12, 15), frequency=2, day_count=DayCount.ACT_ACT_ICMA)
    # 15 Jun -> 15 Dec 2025 = 183 days; settle 15 Sep = 92 days elapsed
    assert b.accrued_interest(date(2025, 9, 15)) == pytest.approx(2.0 * 92 / 183)
