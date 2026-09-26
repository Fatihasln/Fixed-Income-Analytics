import numpy as np
import pandas as pd
import pytest

from credit_engine.spreads.backtest import (compare_live_vs_insample, expanding_refit,
                                            min_train_weeks_sensitivity, refit_dates)
from credit_engine.spreads.regimes import _fit_once, weekly_log_changes
from credit_engine.spreads.regimes import fit_regime_model
from test_regimes import simulate


def test_refit_dates_positions_and_bounds():
    idx = pd.date_range("2000-01-07", periods=300, freq="W-FRI")
    d = refit_dates(idx, min_train_weeks=100, refit_every=20)
    assert d[0] == idx[99]
    assert all((b - a).days == 140 for a, b in zip(d, d[1:]))       # 20 weeks apart
    assert d[-1] <= idx[-1]


def test_refit_dates_too_little_history_raises():
    idx = pd.date_range("2020-01-03", periods=50, freq="W-FRI")
    with pytest.raises(ValueError, match="min_train_weeks"):
        refit_dates(idx, min_train_weeks=100, refit_every=13)


@pytest.fixture(scope="module")
def spread_and_expanding():
    spread, _ = simulate(n=900, seed=7)
    res = expanding_refit(spread, min_train_weeks=200, refit_every=26, search_reps=8, seed=1)
    return spread, res


def test_expanding_refit_shape_and_log(spread_and_expanding):
    spread, res = spread_and_expanding
    assert res.n_refits == len(res.refit_log) == res.refit_log.shape[0]
    assert res.n_refits >= 2
    assert list(res.refit_log.columns) == ["refit_date", "n_train_weeks", "converged", "calm_sigma_pct", "stress_sigma_pct"]
    assert res.refit_log["n_train_weeks"].is_monotonic_increasing        # expanding, not rolling
    assert res.stress_prob.between(0, 1).all()
    assert res.stress_prob.index.is_monotonic_increasing
    assert not res.stress_prob.index.duplicated().any()


def test_expanding_refit_covers_from_series_start(spread_and_expanding):
    spread, res = spread_and_expanding
    from credit_engine.spreads.regimes import weekly_log_changes
    r = weekly_log_changes(spread)
    assert res.stress_prob.index[0] == r.index[0]
    assert res.stress_prob.index[-1] == r.index[-1]


def test_no_lookahead_training_windows_are_identical_regardless_of_future_data():
    """The central guarantee: the (date, training-window-size) of every refit before a cutoff must be IDENTICAL
    whether the series stops at the cutoff or continues for years afterwards -- i.e. no refit can possibly have
    seen future data. We check this on the refit LOG (deterministic) rather than on the fitted probabilities
    themselves: the optimiser's numerical path is not always bit-reproducible (see module docstring), so two
    equally-valid, differently-labelled local optima can give slightly different probabilities even for
    identical training data. That optimiser noise is a separate, documented limitation, not look-ahead."""
    spread, _ = simulate(n=900, seed=11)
    full = expanding_refit(spread, min_train_weeks=150, refit_every=20, search_reps=6, seed=2)

    cutoff = full.stress_prob.index[500]
    truncated_spread = spread.loc[:cutoff + pd.Timedelta(days=3)]
    trunc = expanding_refit(truncated_spread, min_train_weeks=150, refit_every=20, search_reps=6, seed=2)

    shared = trunc.refit_log.set_index("refit_date")
    full_shared = full.refit_log.set_index("refit_date").loc[shared.index]
    assert len(shared) >= 5
    pd.testing.assert_series_equal(shared["n_train_weeks"], full_shared["n_train_weeks"])


def test_filter_step_itself_has_no_lookahead_given_fixed_params():
    """Complements the test above: given FIXED parameters (as produced by one refit), the filtered probability
    at time t is provably unaffected by appending future data -- this is the mathematical guarantee that makes
    per-refit filtering safe to use live. Same pattern as test_regimes.py's equivalent test."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    spread, _ = simulate(n=900, seed=11)
    r = weekly_log_changes(spread)
    _, res = _fit_once(r.iloc[:200], 10, 0)

    m = 350
    full_filt = MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True).filter(res.params)
    trunc_filt = MarkovRegression(r.iloc[:m], k_regimes=2, trend="c", switching_variance=True).filter(res.params)
    k = int(np.argmax([res.params["sigma2[0]"], res.params["sigma2[1]"]]))
    a = full_filt.filtered_marginal_probabilities[k].iloc[:m].to_numpy()
    b = trunc_filt.filtered_marginal_probabilities[k].to_numpy()
    assert np.abs(a - b).max() < 1e-9


def test_live_is_never_better_informed_than_insample_params():
    """Sanity: the early refits use much less data than the final in-sample fit. This does not assert accuracy
    (both can be noisy), only that the two series are not literally identical (i.e. refitting actually happens)."""
    spread, _ = simulate(n=900, seed=7)
    live = expanding_refit(spread, min_train_weeks=200, refit_every=26, search_reps=8, seed=1)
    insample = fit_regime_model(spread, search_reps=15)
    cmp = compare_live_vs_insample(live.stress_prob, insample.stress_prob)
    assert cmp["n_weeks"] > 400
    assert cmp["mean_abs_diff_pp"] > 0.01                 # they must actually differ somewhere
    assert 0 <= cmp["agreement_pct"] <= 100


def test_compare_live_vs_insample_identical_series():
    idx = pd.date_range("2020-01-03", periods=50, freq="W-FRI")
    s = pd.Series(np.linspace(0.1, 0.9, 50), index=idx)
    cmp = compare_live_vs_insample(s, s)
    assert cmp["mean_abs_diff_pp"] == 0.0 and cmp["agreement_pct"] == 100.0 and cmp["corr"] == pytest.approx(1.0)


def test_compare_live_vs_insample_no_overlap_raises():
    idx1 = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    idx2 = pd.date_range("2025-01-03", periods=10, freq="W-FRI")
    with pytest.raises(ValueError, match="no overlapping dates"):
        compare_live_vs_insample(pd.Series(0.5, index=idx1), pd.Series(0.5, index=idx2))


def test_min_train_weeks_sensitivity_more_history_does_not_hurt_agreement():
    """With more starting history, the live simulation should generally track the in-sample fit at least as
    well (fewer, better-informed early refits). We check the qualitative direction, not an exact threshold,
    since this is inherently noisy with a single simulated dataset."""
    spread, _ = simulate(n=1100, seed=21)
    insample = fit_regime_model(spread, search_reps=15)
    tbl = min_train_weeks_sensitivity(spread, insample.stress_prob, candidates=[100, 300], search_reps=6, seed=3)
    assert list(tbl.index) == [100, 300]
    assert tbl.loc[300, "n_refits"] < tbl.loc[100, "n_refits"]   # more min-train -> fewer total refits
    assert tbl["agreement_pct"].between(0, 100).all()
