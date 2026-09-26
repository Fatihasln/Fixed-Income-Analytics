"""Expanding-window refit: simulate what a LIVE system would have seen.

Section E's ``fit_regime_model`` fits parameters on the FULL sample (in-sample). That is fine to learn the
model's structure, but it is optimistic: at any past date t, the in-sample model already "knows" the
parameters that best explain the whole history, including everything after t.

Here the model is refit periodically (e.g. quarterly) using ONLY data available up to each refit date, and the
resulting fixed parameters are used to filter forward until the next refit. Concatenating these pieces gives a
"live" stress-probability series where the value at any date t depends only on parameters estimated from data
strictly before t and on data up to and including t.

NOTE ON REPRODUCIBILITY: the per-refit optimisation (EM + quasi-Newton) is not always bit-identical across
runs even with a fixed seed, because the underlying linear algebra can use multi-threaded BLAS with a
non-deterministic reduction order; this occasionally lands in a different (but statistically equivalent,
label-swapped) local optimum. This does NOT create look-ahead: what is tested and guaranteed is that the
training window for refit i (hence everything the optimiser can possibly see) never includes data at or after
the truncation point of a shorter run of the same series -- see test_backtest.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regimes import CALM, STRESS, RegimeResult, _fit_once, weekly_log_changes


@dataclass(frozen=True)
class ExpandingResult:
    stress_prob: pd.Series      # "live" filtered stress probability, stitched across refits
    refit_log: pd.DataFrame     # one row per refit: date, n_train_weeks, converged, calm/stress sigma, k_stress
    n_refits: int


def refit_dates(index: pd.DatetimeIndex, min_train_weeks: int, refit_every: int) -> list[pd.Timestamp]:
    """Refit dates: index[min_train_weeks - 1], then every ``refit_every`` weeks after, up to the last date.

    Determined purely by position counted forward from the start, so it does NOT depend on how much data
    exists beyond a given point (needed for the no-lookahead guarantee: see test_backtest.py).
    """
    if len(index) <= min_train_weeks:
        raise ValueError(f"only {len(index)} weeks of data, need > {min_train_weeks} (min_train_weeks)")
    return list(index[min_train_weeks - 1 :: refit_every])


def _fit_once_robust(train: pd.Series, search_reps: int, seed: int, max_attempts: int = 8):
    """_fit_once, retried with new random start values if the optimiser hits a numerical singularity
    (rare with very short/degenerate windows) or fails to converge; keeps the best log-likelihood seen."""
    best = None
    for attempt in range(max_attempts):
        reps = max(1, search_reps >> attempt)
        try:
            _, res = _fit_once(train, reps, seed + 1000 * attempt)
        except (np.linalg.LinAlgError, RuntimeError):
            continue
        conv = res.mle_retvals.get("converged") if hasattr(res, "mle_retvals") else None
        if best is None or (bool(conv), res.llf) > (bool(best.mle_retvals.get("converged")), best.llf):
            best = res
        if conv:
            break
    if best is None:
        raise RuntimeError(f"regime fit failed on all {max_attempts} attempts (numerical singularity)")
    return best


def expanding_refit(spread: pd.Series, *, freq: str = "W-FRI", min_train_weeks: int = 104,
                    refit_every: int = 13, search_reps: int = 10, seed: int = 0) -> ExpandingResult:
    """Refit every ``refit_every`` weeks (13 ~ one quarter) on an expanding window; stitch filtered probabilities.

    ``min_train_weeks`` (default 104 ~ 2 years) is the minimum history before the first fit is attempted.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    r = weekly_log_changes(spread, freq)
    dates = refit_dates(r.index, min_train_weeks, refit_every)

    pieces: list[pd.Series] = []
    log_rows = []
    for i, d in enumerate(dates):
        train = r.loc[:d]
        res = _fit_once_robust(train, search_reps, seed + i)
        conv = res.mle_retvals.get("converged") if hasattr(res, "mle_retvals") else None
        sig2 = np.array([res.params["sigma2[0]"], res.params["sigma2[1]"]])
        k_stress = int(np.argmax(sig2))

        window_end = dates[i + 1] if i + 1 < len(dates) else r.index[-1]
        window = r.loc[:window_end]
        filt = MarkovRegression(window, k_regimes=2, trend="c", switching_variance=True).filter(res.params)
        piece_start = r.index[0] if i == 0 else d
        piece = filt.filtered_marginal_probabilities[k_stress].loc[piece_start:window_end]
        pieces.append(piece)
        log_rows.append({"refit_date": d.date(), "n_train_weeks": len(train), "converged": conv,
                         "calm_sigma_pct": float(np.sqrt(sig2[1 - k_stress]) * 100),
                         "stress_sigma_pct": float(np.sqrt(sig2[k_stress]) * 100)})

    stitched = pd.concat(pieces)
    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index().rename("p_stress_live")
    return ExpandingResult(stress_prob=stitched, refit_log=pd.DataFrame(log_rows), n_refits=len(dates))


def compare_live_vs_insample(live: pd.Series, insample: pd.Series) -> dict:
    """How different is the honest 'live' probability from the optimistic in-sample one?

    mean_abs_diff_pp / max_abs_diff_pp: in percentage points of probability.
    agreement_pct: share of weeks where both sides of a 0.5 threshold agree.
    corr: correlation of the two probability series (not just the 0/1 calls).
    """
    df = pd.concat({"live": live, "insample": insample}, axis=1).dropna()
    if df.empty:
        raise ValueError("no overlapping dates between live and in-sample probabilities")
    diff = (df["live"] - df["insample"]).abs()
    agree = ((df["live"] >= 0.5) == (df["insample"] >= 0.5)).mean()
    return {
        "n_weeks": len(df),
        "mean_abs_diff_pp": float(diff.mean() * 100),
        "max_abs_diff_pp": float(diff.max() * 100),
        "agreement_pct": float(agree * 100),
        "corr": float(df["live"].corr(df["insample"])),
    }


def min_train_weeks_sensitivity(spread: pd.Series, insample_stress_prob: pd.Series,
                                candidates: list[int] = (52, 104, 156, 208), *, refit_every: int = 13,
                                search_reps: int = 8, seed: int = 0) -> pd.DataFrame:
    """Re-run the expanding-window simulation for several ``min_train_weeks`` values and compare each to the
    in-sample fit, to check whether the live/in-sample gap (Section 6) shrinks with more starting history."""
    from .backtest import compare_live_vs_insample  # local import: avoids a circular reference at module load

    rows = []
    for w in candidates:
        live = expanding_refit(spread, min_train_weeks=w, refit_every=refit_every, search_reps=search_reps, seed=seed)
        cmp = compare_live_vs_insample(live.stress_prob, insample_stress_prob)
        rows.append({"min_train_weeks": w, "n_refits": live.n_refits, **cmp})
    return pd.DataFrame(rows).set_index("min_train_weeks")
