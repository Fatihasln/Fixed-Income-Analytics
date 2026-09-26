from datetime import date

import pytest

from credit_engine.pricing.bond import Bond, dirty_price, dirty_price_from_curve
from credit_engine.pricing.curves import flat_curve, linear_curve
from credit_engine.risk.cs01 import cs01, dv01_curve, spread_duration, z_spread
from credit_engine.risk.duration import dv01, modified_duration

BOND = Bond(0.06, date(2033, 9, 15))
SETTLE = date(2026, 3, 10)


def test_flat_curve_plus_spread_reproduces_ytm_price():
    y, s = 0.07, 0.025
    assert dirty_price_from_curve(BOND, SETTLE, flat_curve(y - s), s) == pytest.approx(
        dirty_price(BOND, SETTLE, y), abs=1e-10
    )


def test_cs01_equals_dv01_for_parallel_shift_flat_curve():
    y, s = 0.07, 0.025
    assert cs01(BOND, SETTLE, flat_curve(y - s), s) == pytest.approx(dv01(BOND, SETTLE, y), rel=1e-6)


def test_cs01_equals_curve_dv01_even_on_sloped_curve():
    """z and s enter additively -> parallel bumps are identical for a bullet bond."""
    curve = linear_curve([0.5, 2, 5, 10], [0.04, 0.042, 0.045, 0.048])
    s = 0.03
    assert cs01(BOND, SETTLE, curve, s) == pytest.approx(dv01_curve(BOND, SETTLE, curve, s), rel=1e-9)


def test_spread_duration_close_to_modified_duration_of_ytm():
    y, s = 0.07, 0.025
    sd = spread_duration(BOND, SETTLE, flat_curve(y - s), s)
    assert sd == pytest.approx(modified_duration(BOND, SETTLE, y), rel=1e-6)


def test_z_spread_roundtrip():
    curve = linear_curve([0.5, 2, 5, 10], [0.04, 0.042, 0.045, 0.048])
    p = dirty_price_from_curve(BOND, SETTLE, curve, 0.0312)
    assert z_spread(BOND, SETTLE, p, curve) == pytest.approx(0.0312, abs=1e-10)


def test_cs01_scales_with_notional():
    c = flat_curve(0.04)
    assert cs01(BOND, SETTLE, c, 0.02, notional=5e6) == pytest.approx(cs01(BOND, SETTLE, c, 0.02) * 5e4)
