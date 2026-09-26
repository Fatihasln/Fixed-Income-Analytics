import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from credit_engine.data.fred import CATALOG
from credit_engine.spreads.events import EVENTS, plot_with_events
from credit_engine.spreads.spread_calc import find_gaps, gaps_report, proxy_validation, spread


def _bdays(start, n):
    return pd.bdate_range(start, periods=n)


def test_spread_uses_common_dates_only():
    a = pd.Series([0.06, 0.061, np.nan, 0.063], index=_bdays("2024-01-01", 4))
    b = pd.Series([0.04, np.nan, 0.041, 0.042], index=_bdays("2024-01-01", 4))
    s = spread(a, b, "x")
    assert s.name == "x" and list(s.round(4)) == [0.02, 0.021]


def test_catalog_has_moody_yield_levels():
    assert CATALOG["baa_yield"][0] == "DBAA" and CATALOG["aaa_yield"][0] == "DAAA"


def test_find_gaps_ignores_weekends_and_holidays_but_finds_discontinuation():
    idx = pd.bdate_range("1985-01-01", "1986-12-31").append(pd.bdate_range("1993-10-01", "1994-06-30"))
    s = pd.Series(1.0, index=idx)
    s = s.drop(pd.Timestamp("1985-12-25"))                       # a normal holiday: must be ignored
    g = find_gaps(s)
    assert len(g) == 1
    assert g.loc[0, "last_obs_before"].year == 1986 and g.loc[0, "first_obs_after"].year == 1993
    assert g.loc[0, "calendar_days"] > 2000


def test_gaps_report_flags_only_series_with_gaps():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    good = pd.Series(1.0, index=idx)
    holey = good.copy(); holey.loc["2020-05-01":"2020-08-01"] = np.nan
    rep = gaps_report(pd.DataFrame({"good": good, "holey": holey}))
    assert list(rep["series"]) == ["holey"]
    assert gaps_report(pd.DataFrame({"good": good})).empty


def _pair(beta=0.8, offset=0.005, noise=0.0004, n=800, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-09-22", periods=n)
    proxy = pd.Series(0.02 + np.cumsum(rng.normal(0, 0.0008, n)), index=idx)
    ref = beta * proxy - offset + rng.normal(0, noise, n)
    return proxy, pd.Series(ref, index=idx)


def test_proxy_validation_recovers_known_relationship():
    proxy, ref = _pair(beta=0.8, offset=0.005, noise=0.0001)
    r = proxy_validation(proxy, ref)
    # gap = proxy - ref = 0.2*proxy + 0.005 ; mean around 0.2*mean(proxy) + 0.005
    assert r["level_gap_mean_bp"] == pytest.approx((0.2 * proxy.mean() + 0.005) * 1e4, rel=0.02)
    assert r["corr_weekly_changes"] > 0.95
    assert r["beta_weekly"] == pytest.approx(0.8, abs=0.05)
    assert r["n_days"] == 800


def test_proxy_validation_detects_unrelated_series():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2023-09-22", periods=800)
    a = pd.Series(0.02 + np.cumsum(rng.normal(0, 0.0008, 800)), index=idx)
    b = pd.Series(0.01 + np.cumsum(rng.normal(0, 0.0008, 800)), index=idx)
    assert abs(proxy_validation(a, b)["corr_weekly_changes"]) < 0.3


def test_proxy_validation_short_overlap_raises():
    proxy, ref = _pair(n=30)
    with pytest.raises(ValueError, match="overlap too short"):
        proxy_validation(proxy, ref)


def test_plot_with_events_runs_and_skips_out_of_range_events():
    idx = pd.bdate_range("2007-01-01", "2010-12-31")
    s = pd.Series(np.linspace(150, 600, len(idx)), index=idx)
    fig = plot_with_events(s, "test")
    n_spans = sum(1 for p in fig.axes[0].patches)
    in_range = [e for e in EVENTS if pd.Timestamp(e[2]) >= idx[0] and pd.Timestamp(e[1]) <= idx[-1]]
    assert n_spans == len(in_range) and 0 < len(in_range) < len(EVENTS)
