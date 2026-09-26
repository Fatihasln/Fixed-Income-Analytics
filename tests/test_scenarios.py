from datetime import date

import numpy as np
import pytest

from credit_engine.pricing.bond import Bond, dirty_price_from_curve
from credit_engine.pricing.curves import flat_curve
from credit_engine.portfolio.scenarios import (PortfolioBond, historical_style_scenarios, parallel,
                                               price_portfolio, run_scenarios, scaled_historical_scenarios,
                                               spread_widening, steepen)
from credit_engine.risk.duration import dv01

SETTLE = date(2026, 9, 21)
BOND = Bond(0.05, date(2033, 9, 15))
PORT = [PortfolioBond("A", BOND, 0.02, 10_000_000)]


def test_parallel_up_hurts_parallel_down_helps_symmetric_small_shock():
    up = run_scenarios(PORT, SETTLE, flat_curve(0.04), [parallel(1)])
    down = run_scenarios(PORT, SETTLE, flat_curve(0.04), [parallel(-1)])
    assert up["portfolio_pnl_$"].iloc[0] < 0 < down["portfolio_pnl_$"].iloc[0]
    assert up["portfolio_pnl_$"].iloc[0] == pytest.approx(-down["portfolio_pnl_$"].iloc[0], rel=1e-3)  # small shock: near-symmetric


def test_parallel_100bp_matches_dv01_approximately():
    """Exact reprice at 1bp should match the DV01 formula closely (linear regime)."""
    df = run_scenarios(PORT, SETTLE, flat_curve(0.04), [parallel(1)])
    exact = -df["portfolio_pnl_$"].iloc[0]
    approx = dv01(BOND, SETTLE, 0.06, notional=10_000_000)
    assert exact == pytest.approx(approx, rel=0.02)


def test_large_shock_convexity_makes_gain_exceed_loss():
    """At 100bp, convexity means the price gain from a down-shock exceeds the loss from an up-shock
    of the same size -- this is exactly what a linear DV01 estimate would miss."""
    df = run_scenarios(PORT, SETTLE, flat_curve(0.04), [parallel(100), parallel(-100)])
    loss = abs(df.loc["Parallel +100bp", "portfolio_pnl_$"])
    gain = df.loc["Parallel -100bp", "portfolio_pnl_$"]
    assert gain > loss


def test_spread_widening_hurts_and_scales_with_notional():
    df1 = run_scenarios(PORT, SETTLE, flat_curve(0.04), [spread_widening(100)])
    port2 = [PortfolioBond("A", BOND, 0.02, 20_000_000)]
    df2 = run_scenarios(port2, SETTLE, flat_curve(0.04), [spread_widening(100)])
    assert df1["portfolio_pnl_$"].iloc[0] < 0
    assert df2["portfolio_pnl_$"].iloc[0] == pytest.approx(df1["portfolio_pnl_$"].iloc[0] * 2, rel=1e-6)


def test_steepener_pivots_are_exact():
    """At exactly the pivot maturities, the curve shift must equal the requested short/long bp exactly."""
    sc = steepen(-50, 100, pivot_years=(2.0, 30.0))
    shifted = sc.curve_shift(np.array([2.0, 30.0]))
    assert shifted[0] == pytest.approx(-50 / 1e4) and shifted[1] == pytest.approx(100 / 1e4)


def test_per_bond_pnl_sums_to_portfolio_pnl():
    port = [PortfolioBond("A", Bond(0.05, date(2031, 9, 15)), 0.01, 5_000_000),
           PortfolioBond("B", Bond(0.08, date(2036, 9, 15)), 0.04, 8_000_000)]
    df = run_scenarios(port, SETTLE, flat_curve(0.045), [parallel(75)])
    total = df.loc["Parallel +75bp", ["A_pnl_$", "B_pnl_$"]].sum()
    assert total == pytest.approx(df.loc["Parallel +75bp", "portfolio_pnl_$"], rel=1e-8)


def test_price_portfolio_matches_manual_sum():
    port = [PortfolioBond("A", Bond(0.05, date(2031, 9, 15)), 0.01, 5_000_000),
           PortfolioBond("B", Bond(0.08, date(2036, 9, 15)), 0.04, 8_000_000)]
    curve = flat_curve(0.045)
    manual = sum(dirty_price_from_curve(pb.bond, SETTLE, curve, pb.spread) / 100 * pb.notional for pb in port)
    assert price_portfolio(port, SETTLE, curve) == pytest.approx(manual, rel=1e-10)


def test_historical_style_scenarios_widen_by_correct_amount():
    scs = historical_style_scenarios({"2008 GFC": 620, "COVID-19": 430}, current_spread_bp=150)
    assert len(scs) == 2
    by_name = {s.name: s.spread_shift_bp for s in scs}
    assert any(v == pytest.approx(470) for v in by_name.values())   # 620 - 150
    assert any(v == pytest.approx(280) for v in by_name.values())   # 430 - 150


def test_zero_shock_scenario_gives_zero_pnl():
    df = run_scenarios(PORT, SETTLE, flat_curve(0.04), [parallel(0)])
    assert df["portfolio_pnl_$"].iloc[0] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------ per-bond (rating-scaled) spread shifts
def test_shift_for_resolves_uniform_and_per_bond():
    from credit_engine.portfolio.scenarios import _shift_for
    assert _shift_for(50.0, "anything") == 50.0
    assert _shift_for({"A": 10.0, "B": 200.0}, "B") == 200.0
    with pytest.raises(KeyError, match="no spread shift"):
        _shift_for({"A": 10.0}, "C")


def test_scaled_historical_scenarios_widen_by_different_amounts_per_rating():
    """The exact fix for the H2 bug: HY should widen by MORE than AAA for the same historical episode."""
    ratios = {"AAA-like": 0.3, "BBB-like": 1.0, "HY-like": 3.5}
    scs = scaled_historical_scenarios({"2008 GFC": 620.0}, current_spread_bp=110.0, rating_ratios=ratios)
    assert len(scs) == 1
    shift = scs[0].spread_shift_bp
    assert isinstance(shift, dict)
    base = 620.0 - 110.0
    assert shift["AAA-like"] == pytest.approx(base * 0.3)
    assert shift["BBB-like"] == pytest.approx(base * 1.0)
    assert shift["HY-like"] == pytest.approx(base * 3.5)
    assert shift["HY-like"] > shift["BBB-like"] > shift["AAA-like"]


def test_scaled_historical_scenarios_pnl_hurts_hy_more_than_aaa():
    port = [PortfolioBond("AAA-like", Bond(0.045, date(2031, 9, 15)), 0.006, 10_000_000),
           PortfolioBond("HY-like", Bond(0.085, date(2033, 9, 15)), 0.045, 10_000_000)]
    ratios = {"AAA-like": 0.3, "HY-like": 3.5}
    scs = scaled_historical_scenarios({"2008 GFC": 620.0}, current_spread_bp=110.0, rating_ratios=ratios)
    df = run_scenarios(port, SETTLE, flat_curve(0.045), scs)
    aaa_loss = abs(df["AAA-like_pnl_$"].iloc[0])
    hy_loss = abs(df["HY-like_pnl_$"].iloc[0])
    assert hy_loss > aaa_loss   # same notional, but HY widens far more -> bigger loss


def test_scaled_historical_scenarios_missing_bond_ratio_raises_when_priced():
    port = [PortfolioBond("Unlisted", Bond(0.05, date(2031, 9, 15)), 0.02, 10_000_000)]
    scs = scaled_historical_scenarios({"2008 GFC": 620.0}, current_spread_bp=110.0, rating_ratios={"Other": 1.0})
    with pytest.raises(KeyError, match="no spread shift"):
        run_scenarios(port, SETTLE, flat_curve(0.045), scs)
