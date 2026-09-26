import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from credit_engine.spreads.regimes import (CALM, STRESS, compare_regime_counts, detection_table,
                                           fit_regime_model, plot_regimes, regime_periods, weekly_level_bp,
                                           weekly_log_changes)


def simulate(n=1200, p_cc=0.985, p_ss=0.93, seed=3):
    """Weekly log-changes with a hidden 2-state chain; returns (daily-like spread in decimals, true states)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, int)
    for t in range(1, n):
        stay = p_cc if s[t - 1] == 0 else p_ss
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    r = np.where(s == 1, 0.006, -0.001) + np.where(s == 1, 0.09, 0.03) * rng.standard_normal(n)
    idx = pd.date_range("1986-01-10", periods=n, freq="W-FRI")
    spread = pd.Series(0.02 * np.exp(np.cumsum(r)), index=idx)
    return spread, pd.Series(s[1:], index=idx[1:])       # states aligned with the weekly changes


@pytest.fixture(scope="module")
def fitted():
    spread, states = simulate()
    return spread, states, fit_regime_model(spread)


def test_recovers_hidden_regimes(fitted):
    _, states, res = fitted
    pred = (res.stress_prob >= 0.5).astype(int)
    assert (pred.values == states.reindex(pred.index).values).mean() > 0.90
    assert res.converged in (True, None)


def test_stress_is_the_high_volatility_regime(fitted):
    _, _, res = fitted
    s = res.summary
    assert s.loc[STRESS, "weekly_sigma_pct"] > 2 * s.loc[CALM, "weekly_sigma_pct"]
    assert s.loc[STRESS, "expected_duration_weeks"] < s.loc[CALM, "expected_duration_weeks"]
    # what IS identified: the volatilities (true 3% / 9%), the time share (true ~15.5% stress) ...
    assert s.loc[CALM, "weekly_sigma_pct"] == pytest.approx(3.0, rel=0.15)
    assert s.loc[STRESS, "weekly_sigma_pct"] == pytest.approx(9.0, rel=0.15)
    assert 10 < s.loc[STRESS, "share_of_time_pct"] < 25
    # ... whereas the regime MEANS are NOT: with ~190 stress weeks the standard error of the stress mean (~0.66%/wk)
    # is as large as the true mean (0.6%/wk), so even their sign can flip. We deliberately do not assert on them.
    assert s["share_of_time_pct"].sum() == pytest.approx(100.0, abs=1e-6)


def test_transition_rows_sum_to_one(fitted):
    _, _, res = fitted
    assert res.transition.sum(axis=1).round(8).tolist() == [1.0, 1.0]
    assert list(res.transition.index) == [f"from {CALM}", f"from {STRESS}"]


def test_filtered_probability_has_no_lookahead(fitted):
    """Given fixed parameters, the filtered probability at t must not change when future data are appended.
    The smoothed probability DOES change (it looks ahead): this is why only filtered is allowed for signals."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    spread, _, res = fitted
    r = weekly_log_changes(spread)
    m = 700
    trunc = MarkovRegression(r.iloc[:m], k_regimes=2, trend="c", switching_variance=True).filter(res.params)
    k = int(np.argmax([res.params["sigma2[0]"], res.params["sigma2[1]"]]))
    f_trunc = trunc.filtered_marginal_probabilities[k].to_numpy()
    assert np.abs(res.stress_prob.iloc[:m].to_numpy() - f_trunc).max() < 1e-9
    assert np.abs(res.smoothed_stress_prob.iloc[:m].to_numpy() - f_trunc).max() > 0.05


def test_weekly_helpers_and_validation():
    idx = pd.bdate_range("2020-01-01", periods=60)
    s = pd.Series(np.linspace(0.01, 0.02, 60), index=idx)
    lvl = weekly_level_bp(s)
    assert lvl.iloc[0] == pytest.approx(s.loc["2020-01-03"] * 1e4)        # last obs of the first (Friday) week
    assert (lvl.index.dayofweek == 4).all()
    assert weekly_log_changes(s).iloc[0] == pytest.approx(np.log(lvl.iloc[1] / lvl.iloc[0]))
    bad = s.copy(); bad.iloc[10] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        weekly_log_changes(bad)


def test_regime_periods_exact():
    idx = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    p = pd.Series([0.1, 0.6, 0.9, 0.7, 0.2, 0.1, 0.8, 0.3, 0.6, 0.55], index=idx)
    lvl = pd.Series(np.arange(10) * 10.0, index=idx)
    out = regime_periods(p, level_bp=lvl)
    assert list(out["weeks"]) == [3, 1, 2]
    assert out.loc[0, "peak_prob"] == 0.9 and out.loc[0, "peak_spread_bp"] == 30.0
    assert list(regime_periods(p, min_weeks=2)["weeks"]) == [3, 2]
    assert regime_periods(p * 0).empty


def test_detection_table():
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    p = pd.Series(0.05, index=idx)
    p.iloc[12:18] = 0.9                                             # stress from week 12
    events = [("hit", idx[10], idx[20]), ("miss", idx[25], idx[35]), ("outside", "1990-01-01", "1990-06-01")]
    t = detection_table(p, events)
    assert list(t["event"]) == ["hit", "miss"]                       # out-of-sample event skipped
    assert t.loc[0, "first_alert"] == idx[12].date() and t.loc[0, "lag_days"] == 14
    assert pd.isna(t.loc[1, "lag_days"]) and t.loc[1, "weeks_in_stress_pct"] == 0.0


def test_plot_regimes_runs(fitted):
    spread, _, res = fitted
    fig = plot_regimes(weekly_level_bp(spread), res.stress_prob, "t", events=[("e", "1990-01-01", "1991-01-01")])
    assert len(fig.axes) == 2


# --------------------------------------------------------------- standard errors and model comparison
def test_param_se_present_and_positive(fitted):
    _, _, res = fitted
    assert set(res.param_se.index) == set(res.params.index)
    assert (res.param_se > 0).all()


def test_summary_includes_standard_errors(fitted):
    _, _, res = fitted
    for col in ("weekly_sigma_se_pct", "mean_se_pct"):
        assert col in res.summary.columns
    assert (res.summary["weekly_sigma_se_pct"] > 0).all()


def test_regime_means_are_statistically_weak_evidenced_by_se(fitted):
    """Quantifies the earlier qualitative claim: the regime MEANS are not reliably identified because their
    standard error is comparable to (or larger than) the estimate itself, unlike the volatilities."""
    _, _, res = fitted
    s = res.summary
    mean_t_stats = (s["mean_weekly_change_pct"] / s["mean_se_pct"]).abs()
    sigma_t_stats = (s["weekly_sigma_pct"] / s["weekly_sigma_se_pct"]).abs()
    assert (mean_t_stats < 2).any()          # at least one regime mean is NOT significant at ~95%
    assert (sigma_t_stats > 5).all()         # both volatilities ARE precisely estimated


def test_compare_regime_counts_shape_and_monotonic_loglik():
    spread, _ = simulate(n=900, seed=17)
    tbl = compare_regime_counts(spread, k_range=(2, 3), search_reps=8)
    assert list(tbl.index) == [2, 3]
    assert {"n_params", "loglik", "aic", "bic", "converged"}.issubset(tbl.columns)
    assert tbl.loc[3, "n_params"] > tbl.loc[2, "n_params"]
    assert tbl.loc[3, "loglik"] >= tbl.loc[2, "loglik"] - 1e-6   # more params -> loglik can't get worse
    assert tbl["bic_favors"].iloc[0] in (2, 3) and tbl["aic_favors"].iloc[0] in (2, 3)


def test_compare_regime_counts_bic_penalizes_extra_params_more_than_aic():
    """BIC's penalty (k*ln(n)) exceeds AIC's (2k) whenever n > 7, which always holds here (weekly data over
    even a few years), so BIC's gap between the two models should be at least as large as AIC's gap."""
    spread, _ = simulate(n=900, seed=17)
    tbl = compare_regime_counts(spread, k_range=(2, 3), search_reps=8)
    daic = abs(tbl.loc[3, "aic"] - tbl.loc[2, "aic"])
    dbic = abs(tbl.loc[3, "bic"] - tbl.loc[2, "bic"])
    extra_params = tbl.loc[3, "n_params"] - tbl.loc[2, "n_params"]
    n = 900 - 1
    assert dbic - daic == pytest.approx(extra_params * (np.log(n) - 2), rel=0.05)
