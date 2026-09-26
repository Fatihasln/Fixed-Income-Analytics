from datetime import date

import pytest

from credit_engine.pricing.bond import Bond, clean_price
from credit_engine.pricing.yield_solver import solve_ytm, ytm_brent, ytm_newton

CASES = [
    (Bond(0.05, date(2032, 3, 1)), date(2026, 5, 20)),
    (Bond(0.085, date(2029, 11, 15)), date(2026, 1, 3)),
    (Bond(0.02, date(2055, 8, 31), frequency=1), date(2026, 2, 10)),
    (Bond(0.0, date(2030, 6, 15)), date(2026, 6, 20)),
]


@pytest.mark.parametrize("bond,settle", CASES)
@pytest.mark.parametrize("y", [-0.01, 0.0, 0.03, 0.08, 0.20])
def test_roundtrip_price_to_yield(bond, settle, y):
    p = clean_price(bond, settle, y)
    assert ytm_brent(bond, settle, p) == pytest.approx(y, abs=1e-10)
    assert ytm_newton(bond, settle, p) == pytest.approx(y, abs=1e-9)


def test_brent_and_newton_agree():
    b, s = CASES[1]
    assert solve_ytm(b, s, 92.5, method="brent") == pytest.approx(solve_ytm(b, s, 92.5, method="newton"), abs=1e-10)


def test_dirty_price_input():
    b, s = CASES[0]
    y = 0.045
    dirty = clean_price(b, s, y) + b.accrued_interest(s)
    assert ytm_brent(b, s, dirty, price_type="dirty") == pytest.approx(y, abs=1e-10)


def test_impossible_price_raises():
    b, s = CASES[0]
    with pytest.raises(ValueError):
        ytm_brent(b, s, 1e-9)
    with pytest.raises(ValueError):
        ytm_brent(b, s, -5.0)
