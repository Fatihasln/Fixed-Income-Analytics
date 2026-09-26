from datetime import date

import numpy as np
import pandas as pd
import pytest

from credit_engine.pricing.bond import Bond
from credit_engine.pricing.curves import flat_curve
from credit_engine.portfolio.risk_contribution import (RATING_TO_ICE_BUCKET, factor_correlation,
                                                       realized_weekly_vol_bp, rating_vol_ratio, report_table,
                                                       risk_contribution_report, spread_series_for_rating)
from credit_engine.risk.cs01 import cs01
from credit_engine.risk.duration import dv01

BOND = Bond(0.06, date(2033, 9, 15))
SETTLE = date(2026, 9, 21)


def _series(n, start, vol, seed):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(0.04 + np.cumsum(rng.normal(0, vol, n)), index=idx)


def test_realized_vol_matches_known_random_walk_std():
    """A pure random walk's weekly-change std should recover the daily-vol * sqrt(5) scaling within noise."""
    daily_vol = 0.0006
    s = _series(1000, "2020-01-01", daily_vol, seed=1)
    weekly_vol_bp = realized_weekly_vol_bp(s)
    expected_bp = daily_vol * np.sqrt(5) * 1e4
    assert weekly_vol_bp == pytest.approx(expected_bp, rel=0.25)   # loose: single random draw


def test_realized_vol_requires_enough_data():
    short = pd.Series([0.04, 0.041, 0.042], index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="at least 10"):
        realized_weekly_vol_bp(short)


def test_dv01_equals_cs01_even_though_contributions_differ():
    """The Section 2.3 finding, made explicit: sensitivities are identical, contributions are not."""
    rate = _series(500, "2023-01-01", 0.0004, seed=2)     # low rate vol
    spread = _series(500, "2023-01-01", 0.0020, seed=3)   # high spread vol (HY-like)
    r = risk_contribution_report("test", BOND, SETTLE, flat_curve(0.045), 0.03, 10_000_000, rate, spread)
    assert r.dv01 == pytest.approx(r.cs01, rel=1e-6)
    assert r.spread_risk_contribution > r.rate_risk_contribution
    assert r.dominant_factor == "spread"
    # sanity vs the raw functions
    assert r.dv01 == pytest.approx(dv01(BOND, SETTLE, 0.045 + 0.03, notional=10_000_000), rel=1e-6)
    assert r.cs01 == pytest.approx(cs01(BOND, SETTLE, flat_curve(0.045), 0.03, notional=10_000_000), rel=1e-6)


def test_low_spread_vol_makes_rate_dominant():
    """Flip the volatilities: now rate should dominate (e.g. an AAA-like bond in a calm credit market)."""
    rate = _series(500, "2023-01-01", 0.0020, seed=4)     # high rate vol
    spread = _series(500, "2023-01-01", 0.0003, seed=5)   # very low spread vol (AAA-like, calm)
    r = risk_contribution_report("test", BOND, SETTLE, flat_curve(0.045), 0.006, 10_000_000, rate, spread)
    assert r.dominant_factor == "rate"
    assert r.rate_risk_contribution > r.spread_risk_contribution


def test_realized_vol_window_years_excludes_old_data():
    """A volatility spike far in the past must NOT affect a windowed calculation, only the unwindowed one."""
    calm = _series(600, "2010-01-01", 0.0003, seed=8)
    spike = _series(50, "2015-01-01", 0.02, seed=9)     # a short, loud burst well before the recent window
    tail = _series(300, "2024-01-01", 0.0003, seed=10)   # calm again, recent
    s = pd.concat([calm, spike, tail])
    full = realized_weekly_vol_bp(s)
    windowed = realized_weekly_vol_bp(s, window_years=1.5)
    assert windowed < full / 3          # the old spike inflates the unwindowed number a lot


def test_window_years_none_uses_full_history():
    s = _series(500, "2015-01-01", 0.001, seed=11)
    assert realized_weekly_vol_bp(s, window_years=None) == pytest.approx(realized_weekly_vol_bp(s))


def test_per_rating_spread_series_breaks_the_identical_share_bug():
    """This is the exact bug found from the real notebook run: feeding every bond the SAME spread series makes
    spread_share_pct identical across all bonds (since DV01==CS01 cancels out, leaving only the two shared
    volatilities). Using a rating-appropriate series must break that."""
    rate = _series(700, "2023-06-01", 0.0004, seed=12)
    buckets = {"aaa_oas": _series(700, "2023-06-01", 0.0001, seed=13),
              "bbb_oas": _series(700, "2023-06-01", 0.0003, seed=14),
              "hy_oas": _series(700, "2023-06-01", 0.0018, seed=15)}
    specs = [("AAA-like", "AAA", 0.006), ("BBB-like", "BBB", 0.015), ("HY-like", "BB (HY)", 0.045)]
    reports = [risk_contribution_report(label, BOND, SETTLE, flat_curve(0.045), spr, 10_000_000,
                                        rate_series=rate, spread_series=spread_series_for_rating(rating, buckets))
              for label, rating, spr in specs]
    shares = [r.spread_risk_contribution / (r.rate_risk_contribution + r.spread_risk_contribution) for r in reports]
    assert shares[2] > shares[1] > shares[0]                 # HY > BBB > AAA, strictly increasing
    assert reports[2].dominant_factor == "spread"
    assert reports[0].dominant_factor == "rate"


def test_spread_series_for_rating_mapping():
    buckets = {"aaa_oas": pd.Series([1]), "bbb_oas": pd.Series([2]), "hy_oas": pd.Series([3])}
    assert spread_series_for_rating("AA", buckets).iloc[0] == 1      # maps to aaa_oas
    assert spread_series_for_rating("BBB", buckets).iloc[0] == 2
    assert spread_series_for_rating("B (HY)", buckets).iloc[0] == 3
    with pytest.raises(KeyError, match="unknown rating"):
        spread_series_for_rating("NR", buckets)
    with pytest.raises(KeyError, match="missing from series_by_bucket"):
        spread_series_for_rating("AAA", {"bbb_oas": pd.Series([2])})


def test_all_ratings_are_mapped_to_a_valid_bucket():
    assert set(RATING_TO_ICE_BUCKET.values()) == {"aaa_oas", "bbb_oas", "hy_oas"}


def test_report_table_shape_and_spread_share():
    rate = _series(500, "2023-01-01", 0.0004, seed=6)
    spread = _series(500, "2023-01-01", 0.0020, seed=7)
    r1 = risk_contribution_report("A", BOND, SETTLE, flat_curve(0.045), 0.01, 5_000_000, rate, spread)
    r2 = risk_contribution_report("B", BOND, SETTLE, flat_curve(0.045), 0.05, 5_000_000, rate, spread)
    tbl = report_table([r1, r2])
    assert list(tbl.index) == ["A", "B"]
    assert (tbl["spread_share_pct"] > 50).all()   # spread vol >> rate vol in this fixture
    assert tbl["spread_share_pct"].between(0, 100).all()


# --------------------------------------------------------- correlation and total risk (portfolio variance)
def _correlated_pair(rho_target, n=1000, vol_a=0.0005, vol_b=0.0015, seed=42):
    """Two random walks whose WEEKLY CHANGES have (approximately) the given correlation."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    z1 = rng.normal(0, 1, n)
    z2 = rho_target * z1 + np.sqrt(max(0.0, 1 - rho_target**2)) * rng.normal(0, 1, n)
    a = pd.Series(0.04 + np.cumsum(z1 * vol_a), index=idx)
    b = pd.Series(0.04 + np.cumsum(z2 * vol_b), index=idx)
    return a, b


def test_factor_correlation_recovers_known_sign_and_magnitude():
    neg_rate, neg_spread = _correlated_pair(-0.7, seed=1)
    pos_rate, pos_spread = _correlated_pair(0.7, seed=2)
    r_neg = factor_correlation(neg_rate, neg_spread)
    r_pos = factor_correlation(pos_rate, pos_spread)
    assert r_neg < -0.4                       # flight-to-quality-style: strongly negative recovered
    assert r_pos > 0.4
    assert -1.0 <= r_neg <= 1.0 and -1.0 <= r_pos <= 1.0


def test_factor_correlation_uses_changes_not_levels():
    """Two series that trend in the same direction (correlated LEVELS) but have independent week-to-week
    changes should show a correlation close to zero, not close to one."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=1000)
    a = pd.Series(0.04 + 0.00002 * np.arange(1000) + np.cumsum(rng.normal(0, 0.0005, 1000)), index=idx)
    b = pd.Series(0.03 + 0.00002 * np.arange(1000) + np.cumsum(rng.normal(0, 0.0005, 1000)), index=idx)
    assert a.corr(b) > 0.9                    # levels: strongly correlated (shared trend)
    assert abs(factor_correlation(a, b)) < 0.3  # changes: much weaker, as the two walks are independent


def test_factor_correlation_requires_enough_overlap():
    a = pd.Series([0.04] * 5, index=pd.bdate_range("2024-01-01", periods=5))
    b = pd.Series([0.02] * 5, index=pd.bdate_range("2024-01-01", periods=5))
    with pytest.raises(ValueError, match="overlapping"):
        factor_correlation(a, b)


def test_total_risk_matches_manual_variance_formula():
    rate, spread = _correlated_pair(-0.5, seed=5)
    r = risk_contribution_report("test", BOND, SETTLE, flat_curve(0.045), 0.03, 10_000_000, rate, spread)
    manual_var = r.dv01**2 * r.rate_vol_bp**2 + r.cs01**2 * r.spread_vol_bp**2 \
        + 2 * r.dv01 * r.cs01 * r.rho * r.rate_vol_bp * r.spread_vol_bp
    assert r.total_risk_1sd == pytest.approx(np.sqrt(manual_var), rel=1e-9)


def test_negative_correlation_makes_total_less_than_naive_sum():
    """The whole point of the fix: with rates and spreads moving in OPPOSITE directions (rho < 0), the true
    portfolio risk is LESS than just adding the two marginal contributions (some of the risk hedges itself)."""
    rate, spread = _correlated_pair(-0.6, seed=6)
    r = risk_contribution_report("test", BOND, SETTLE, flat_curve(0.045), 0.03, 10_000_000, rate, spread)
    assert r.rho < 0
    assert r.total_risk_1sd < r.naive_sum_1sd


def test_positive_correlation_makes_total_exceed_marginal_but_not_naive_sum():
    """With rho > 0 the true total risk should be BIGGER than either marginal piece alone, but still capped:
    sqrt(a^2+b^2+2ab*rho) <= a+b for any rho <= 1, so total_risk_1sd never exceeds naive_sum_1sd."""
    rate, spread = _correlated_pair(0.8, seed=7)
    r = risk_contribution_report("test", BOND, SETTLE, flat_curve(0.045), 0.03, 10_000_000, rate, spread)
    assert r.rho > 0
    assert r.rate_risk_contribution < r.total_risk_1sd <= r.naive_sum_1sd


def test_zero_correlation_total_equals_pythagorean_sum():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2022-01-03", periods=2000)
    rate = pd.Series(0.04 + np.cumsum(rng.normal(0, 0.0006, 2000)), index=idx)
    spread = pd.Series(0.04 + np.cumsum(rng.normal(0, 0.0018, 2000)), index=idx)
    r = risk_contribution_report("test", BOND, SETTLE, flat_curve(0.045), 0.03, 10_000_000, rate, spread)
    assert abs(r.rho) < 0.15   # independent random walks: near-zero by construction
    expected = np.sqrt(r.rate_risk_contribution**2 + r.spread_risk_contribution**2)
    assert r.total_risk_1sd == pytest.approx(expected, rel=0.05)


def test_dv01_uses_real_curve_not_just_flat_ytm():
    """With a sloped curve, DV01 must come from an actual curve bump (risk.cs01.dv01_curve), matching CS01's
    own curve-based methodology -- not a flat-YTM approximation that ignores the slope."""
    from credit_engine.pricing.curves import linear_curve
    from credit_engine.risk.cs01 import dv01_curve

    sloped = linear_curve([0.5, 2, 5, 10, 30], [0.04, 0.042, 0.045, 0.048, 0.05])
    rate = _series(500, "2023-01-01", 0.0004, seed=9)
    spread = _series(500, "2023-01-01", 0.0015, seed=10)
    r = risk_contribution_report("test", BOND, SETTLE, sloped, 0.02, 10_000_000, rate, spread)
    expected_dv01 = dv01_curve(BOND, SETTLE, sloped, 0.02, notional=10_000_000)
    assert r.dv01 == pytest.approx(expected_dv01, rel=1e-6)


def test_rating_vol_ratio_orders_hy_above_aaa():
    reference = _series(700, "2023-01-01", 0.001, seed=11)
    buckets = {"aaa_oas": _series(700, "2023-01-01", 0.0003, seed=12),
              "bbb_oas": _series(700, "2023-01-01", 0.0008, seed=13),
              "hy_oas": _series(700, "2023-01-01", 0.003, seed=14)}
    r_aaa = rating_vol_ratio("AAA", buckets, reference)
    r_bbb = rating_vol_ratio("BBB", buckets, reference)
    r_hy = rating_vol_ratio("BB (HY)", buckets, reference)
    assert r_aaa < r_bbb < r_hy
    assert r_aaa < 1.0 < r_hy   # AAA less volatile, HY more volatile than the broad reference


def test_report_table_includes_new_columns():
    rate = _series(500, "2023-01-01", 0.0004, seed=15)
    spread = _series(500, "2023-01-01", 0.0015, seed=16)
    r = risk_contribution_report("A", BOND, SETTLE, flat_curve(0.045), 0.02, 5_000_000, rate, spread)
    tbl = report_table([r])
    for col in ("rho", "naive_sum_1sd_$", "total_risk_1sd_$"):
        assert col in tbl.columns
