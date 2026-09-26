"""Duration tests: hand calculation, textbook identities, analytic vs bump."""
from datetime import date

import pytest

from credit_engine.pricing.bond import Bond, dirty_price
from credit_engine.risk.duration import (
    convexity, dv01, effective_convexity, effective_duration,
    macaulay_duration, modified_duration, price_change_estimate,
)


def test_hand_calculation_2y_5pct_annual_par():
    """2y, 5% annual coupon, y = 5%, settle on a coupon date. Worked by hand:

        PV1 = 5/1.05 = 4.761905        PV2 = 105/1.05^2 = 95.238095   (P = 100)
        D_mac = (1*4.761905 + 2*95.238095)/100 = 1.952381
        D_mod = 1.952381/1.05 = 1.859410
        DV01  = 1.859410 * 100 * 1e-4 = 0.018594
        C     = (5*1*2/1.05^3 + 105*2*3/1.05^4)/100 = 5.269409
    """
    b = Bond(0.05, date(2027, 1, 15), frequency=1)
    s = date(2025, 1, 15)
    assert dirty_price(b, s, 0.05) == pytest.approx(100.0)
    assert macaulay_duration(b, s, 0.05) == pytest.approx(1.952381, abs=1e-6)
    assert modified_duration(b, s, 0.05) == pytest.approx(1.859410, abs=1e-6)
    assert dv01(b, s, 0.05) == pytest.approx(0.018594, abs=1e-6)
    assert convexity(b, s, 0.05) == pytest.approx(5.269409, abs=1e-6)


def test_zero_coupon_macaulay_equals_maturity():
    b = Bond(0.0, date(2030, 6, 15), frequency=2)
    s = date(2025, 6, 15)
    assert macaulay_duration(b, s, 0.07) == pytest.approx(5.0)


def test_duration_ordering():
    """Higher coupon -> lower duration; higher yield -> lower duration."""
    s = date(2026, 1, 15)
    m = date(2036, 1, 15)
    low = macaulay_duration(Bond(0.02, m), s, 0.05)
    high = macaulay_duration(Bond(0.08, m), s, 0.05)
    assert low > high
    assert macaulay_duration(Bond(0.05, m), s, 0.02) > macaulay_duration(Bond(0.05, m), s, 0.10)
    assert macaulay_duration(Bond(0.05, m), s, 0.05) < 10.0   # below maturity for coupon bond


CASES = [
    (Bond(0.05, date(2032, 3, 1)), date(2026, 5, 20), 0.045),
    (Bond(0.085, date(2029, 11, 15)), date(2026, 1, 3), 0.10),
    (Bond(0.03, date(2056, 8, 31), frequency=1), date(2026, 2, 10), 0.055),
]


@pytest.mark.parametrize("bond,settle,y", CASES)
def test_analytic_equals_bump_and_reprice(bond, settle, y):
    # central difference has O(h^2) truncation error, larger for long bonds (30y ~ 1e-6 rel)
    assert effective_duration(bond, settle, y) == pytest.approx(modified_duration(bond, settle, y), rel=1e-5)
    assert effective_convexity(bond, settle, y) == pytest.approx(convexity(bond, settle, y), rel=1e-4)


@pytest.mark.parametrize("bond,settle,y", CASES)
def test_taylor_error_is_third_order(bond, settle, y):
    """With the correct convexity the 2nd-order Taylor error shrinks like dy^3
    (relative error like dy^2): halving the shock cuts the relative error ~4x.
    A wrong convexity would give only ~2x (or no improvement over duration alone)."""
    p0 = dirty_price(bond, settle, y)
    md, cv = modified_duration(bond, settle, y), convexity(bond, settle, y)

    def rel_err(dy, use_convexity=True):
        exact = dirty_price(bond, settle, y + dy) - p0
        est = price_change_estimate(md, cv if use_convexity else 0.0, p0, dy)
        return abs(est - exact) / abs(exact)

    assert rel_err(0.01) < rel_err(0.01, use_convexity=False)        # convexity helps
    ratio = rel_err(0.01) / rel_err(0.005)
    assert 3.0 < ratio < 5.0
    assert rel_err(0.005) < 0.01                                      # <1% at 50bp even for 30y


def test_dv01_scales_with_notional():
    b, s, y = CASES[0]
    assert dv01(b, s, y, notional=10_000_000) == pytest.approx(dv01(b, s, y) * 100_000)
