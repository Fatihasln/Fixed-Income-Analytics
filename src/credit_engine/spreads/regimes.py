"""Two-regime Markov switching model for credit spreads (calm vs stress).

Model: weekly log-changes of the spread, r_t = mu_s + sigma_s * eps_t, with the hidden state s in {calm, stress}
following a Markov chain. Both the mean AND the variance switch. Spread changes are close to stationary, so this is
statistically far better behaved than fitting the (very persistent) spread level; regimes are identified mainly
through volatility clustering, which is what distinguishes stress episodes. The regime MEANS are statistically
weak (their standard errors are about as large as the means themselves); interpret sigma, durations and time shares.

LOOK-AHEAD (the point of this module): the model returns two probability series.
  * ``stress_prob``           FILTERED  P(stress_t | data up to t)      -> usable live, use this for any signal
  * ``smoothed_stress_prob``  SMOOTHED  P(stress_t | ALL data)          -> uses the future, diagnostics only
Note the parameters are still estimated on the full sample (in-sample). A truly live system must refit on an
expanding window; that is a later step. What is guaranteed here: given the parameters, ``stress_prob`` at time t
never changes when future data are appended (tested).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

CALM, STRESS = "calm", "stress"


def weekly_level_bp(spread: pd.Series, freq: str = "W-FRI") -> pd.Series:
    """Weekly (last observation) spread level in bp from a daily spread in decimals."""
    return (spread.dropna().resample(freq).last().dropna() * 1e4).rename("spread_bp")


def weekly_log_changes(spread: pd.Series, freq: str = "W-FRI") -> pd.Series:
    """Weekly log-changes of a strictly positive spread (decimals in)."""
    s = spread.dropna()
    if (s <= 0).any():
        raise ValueError(f"spread must be strictly positive to take logs (min = {s.min():.6f})")
    w = s.resample(freq).last().dropna()
    return np.log(w).diff().dropna().rename("dlog_spread")


@dataclass(frozen=True)
class RegimeResult:
    stress_prob: pd.Series            # FILTERED: safe to use live
    smoothed_stress_prob: pd.Series   # SMOOTHED: look-ahead, diagnostics only
    summary: pd.DataFrame             # one row per regime
    transition: pd.DataFrame          # P(next | current); rows = from, cols = to
    loglik: float
    aic: float
    bic: float
    converged: bool | None
    params: pd.Series
    param_se: pd.Series               # standard errors from the MLE (statsmodels' res.bse), same index as params
    n_obs: int


def _fit_once(r: pd.Series, search_reps: int, seed: int, k_regimes: int = 2):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    np.random.seed(seed)                                   # statsmodels draws start values from the global RNG
    mod = MarkovRegression(r, k_regimes=k_regimes, trend="c", switching_variance=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(search_reps=search_reps, disp=False)
    return mod, res


def _fit_with_retries(r: pd.Series, search_reps: int, seed: int, k_regimes: int = 2, max_attempts: int = 8):
    """Fit with a shrinking random search on failure (rare numerical singularities); keeps the best log-likelihood
    seen and prefers a converged result. Shared by ``fit_regime_model`` and ``compare_regime_counts``."""
    best = None
    for attempt in range(max_attempts):
        reps = max(1, search_reps >> attempt)              # shrink the random search on later, more conservative tries
        try:
            _, res = _fit_once(r, reps, seed + 97 * attempt, k_regimes=k_regimes)   # large odd stride avoids seed collisions
        except (np.linalg.LinAlgError, RuntimeError):
            continue                                       # rare numerical singularity; retry with a new seed
        conv = res.mle_retvals.get("converged") if hasattr(res, "mle_retvals") else None
        if best is None or (bool(conv), res.llf) > (bool(best[1]), best[0].llf):
            best = (res, conv)
        if conv:
            break
    if best is None:
        raise RuntimeError(f"regime fit failed on all {max_attempts} attempts (numerical singularity)")
    return best


def fit_regime_model(spread: pd.Series, *, freq: str = "W-FRI", search_reps: int = 20, seed: int = 0,
                     max_attempts: int = 8) -> RegimeResult:
    """Fit the 2-regime model. Retries with new seeds if the optimiser does not converge; keeps the best log-likelihood."""
    r = weekly_log_changes(spread, freq)
    level = weekly_level_bp(spread, freq).reindex(r.index)
    res, conv = _fit_with_retries(r, search_reps, seed, k_regimes=2, max_attempts=max_attempts)

    sig2 = np.array([res.params["sigma2[0]"], res.params["sigma2[1]"]])
    k_stress = int(np.argmax(sig2))                        # label switching: stress = the high-variance state
    k_calm = 1 - k_stress
    filt = res.filtered_marginal_probabilities
    smooth = res.smoothed_marginal_probabilities
    dur = res.expected_durations
    P = np.asarray(res.regime_transition)[:, :, 0]         # P[i, j] = P(S_t = i | S_{t-1} = j)

    rows = {}
    for name, k in ((CALM, k_calm), (STRESS, k_stress)):
        w = filt[k].to_numpy()
        sigma_se = res.bse[f"sigma2[{k}]"] / (2 * np.sqrt(sig2[k])) * 100   # delta method: se(sqrt(x)) ~ se(x)/(2*sqrt(x))
        rows[name] = {
            "weekly_sigma_pct": float(np.sqrt(sig2[k]) * 100),
            "weekly_sigma_se_pct": float(sigma_se),
            "mean_weekly_change_pct": float(res.params[f"const[{k}]"] * 100),
            "mean_se_pct": float(res.bse[f"const[{k}]"] * 100),
            "expected_duration_weeks": float(dur[k]),
            "avg_spread_bp": float(np.average(level.to_numpy(), weights=w)),
            "share_of_time_pct": float(w.mean() * 100),
        }
    order = [k_calm, k_stress]
    trans = pd.DataFrame(P.T[np.ix_(order, order)], index=[f"from {CALM}", f"from {STRESS}"],
                         columns=[f"to {CALM}", f"to {STRESS}"])
    return RegimeResult(
        stress_prob=filt[k_stress].rename("p_stress_filtered"),
        smoothed_stress_prob=smooth[k_stress].rename("p_stress_smoothed"),
        summary=pd.DataFrame(rows).T,
        transition=trans,
        loglik=float(res.llf), aic=float(res.aic), bic=float(res.bic),
        converged=None if conv is None else bool(conv),
        params=res.params, param_se=res.bse, n_obs=len(r),
    )


def regime_periods(prob: pd.Series, threshold: float = 0.5, level_bp: pd.Series | None = None,
                   min_weeks: int = 1) -> pd.DataFrame:
    """Contiguous runs where prob >= threshold: start, end, number of weeks, peak probability (and peak spread)."""
    flag = (prob >= threshold).to_numpy()
    rows, i, n = [], 0, len(prob)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1]:
            j += 1
        seg = prob.iloc[i:j + 1]
        row = {"start": seg.index[0].date(), "end": seg.index[-1].date(), "weeks": j - i + 1,
               "peak_prob": float(seg.max())}
        if level_bp is not None:
            row["peak_spread_bp"] = float(level_bp.reindex(seg.index).max())
        rows.append(row)
        i = j + 1
    cols = ["start", "end", "weeks", "peak_prob"] + (["peak_spread_bp"] if level_bp is not None else [])
    out = pd.DataFrame(rows, columns=cols)
    return out[out["weeks"] >= min_weeks].reset_index(drop=True)


def detection_table(prob: pd.Series, events, threshold: float = 0.5, pad_days: int = 0) -> pd.DataFrame:
    """For each (name, start, end) event: did the FILTERED stress probability cross the threshold, and when?

    Event dates are approximate/subjective, so ``lag_days`` (first alert minus event start) is indicative, not a
    rigorous detection delay. Events outside the sample are skipped.
    """
    rows = []
    for name, a, b in events:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b < prob.index[0] or a > prob.index[-1]:
            continue
        win = prob.loc[a - pd.Timedelta(days=pad_days):b]
        if win.empty:
            continue
        hit = win[win >= threshold]
        rows.append({
            "event": name, "window": f"{a.date()} -> {b.date()}",
            "first_alert": hit.index[0].date() if len(hit) else None,
            "lag_days": (hit.index[0] - a).days if len(hit) else np.nan,
            "weeks_in_stress_pct": float((win >= threshold).mean() * 100),
            "peak_prob": float(win.max()),
        })
    return pd.DataFrame(rows)


def plot_regimes(spread_bp: pd.Series, stress_prob: pd.Series, title: str, events=None, threshold: float = 0.5,
                 figsize=(12, 7)):
    """Top: spread with marked events. Bottom: model's FILTERED stress probability with the same events."""
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(spread_bp.index, spread_bp.values, lw=1.0, color="tab:blue")
    a1.set_ylabel("bp")
    a1.set_title(title)
    a2.fill_between(stress_prob.index, stress_prob.values, color="tab:orange", alpha=0.6, lw=0)
    a2.axhline(threshold, color="k", lw=0.8, ls="--")
    a2.set_ylim(0, 1)
    a2.set_ylabel("P(stress), filtered")
    if events:
        top = float(spread_bp.max())
        for name, a, b in events:
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            if b < spread_bp.index[0] or a > spread_bp.index[-1]:
                continue
            for ax in (a1, a2):
                ax.axvspan(a, b, color="tab:red", alpha=0.15, lw=0)
            a1.text(a + (b - a) / 2, top, name, rotation=90, va="top", ha="center", fontsize=7, color="darkred")
    fig.tight_layout()
    return fig


def compare_regime_counts(spread: pd.Series, *, freq: str = "W-FRI", k_range: tuple[int, ...] = (2, 3),
                          search_reps: int = 15, seed: int = 0, max_attempts: int = 8) -> pd.DataFrame:
    """Fit the same weekly-log-change series with different numbers of hidden regimes and compare by
    log-likelihood, AIC and BIC (lower AIC/BIC is better; BIC penalises extra parameters more heavily and is
    the more conservative choice for "how many regimes are actually supported by the data").

    A 3-regime model has more parameters (more ways to fit noise) and will always have a higher (or equal)
    log-likelihood than a 2-regime model on the same data; AIC/BIC exist precisely to penalise that extra
    flexibility. If AIC/BIC do NOT improve when moving from 2 to 3 regimes, that is evidence the second regime
    the 3-state model finds is not earning its extra parameters, not evidence that credit spreads only ever
    have two "true" states.
    """
    r = weekly_log_changes(spread, freq)
    rows = []
    for k in k_range:
        res, conv = _fit_with_retries(r, search_reps, seed, k_regimes=k, max_attempts=max_attempts)
        n_params = len(res.params)
        rows.append({"k_regimes": k, "n_params": n_params, "loglik": float(res.llf),
                    "aic": float(res.aic), "bic": float(res.bic), "converged": conv})
    out = pd.DataFrame(rows).set_index("k_regimes")
    out["bic_favors"] = out["bic"].idxmin()
    out["aic_favors"] = out["aic"].idxmin()
    return out
