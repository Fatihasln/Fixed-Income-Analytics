"""Risk contribution: sensitivity x typical factor move, and total risk WITH correlation.

Section 2.3's finding was that CS01 equals curve DV01 exactly for a parallel shift on a plain bullet bond
(tested in test_cs01.py). That means the two risk MEASURES cannot be compared directly -- what differs between
rate risk and credit risk is how much each FACTOR typically moves, not the bond's sensitivity to it.

This module makes that comparison explicit and quantitative:

    weekly_risk_contribution = sensitivity_per_bp * typical_weekly_move_in_bp

where "typical move" is the realized weekly standard deviation of the factor (Treasury yield for rate risk,
credit spread for spread risk), estimated from real FRED data -- not assumed.

IMPORTANT -- correlation: ``rate_risk_contribution`` and ``spread_risk_contribution`` are MARGINAL pieces; they
do not sum to the portfolio's true P&L variance unless rate and spread moves are uncorrelated. In reality they
usually are not (e.g. flight-to-quality: rates often fall while spreads widen in a crisis, a NEGATIVE
correlation). The correct total is

    Var(P&L) = DV01^2 * sigma_r^2 + CS01^2 * sigma_s^2 + 2 * DV01 * CS01 * rho * sigma_r * sigma_s

which is what ``total_risk_1sd`` computes (a one-standard-deviation $ P&L swing in a typical week). Ignoring
rho and just adding the two marginal contributions overstates risk when rho < 0 and understates it when rho > 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..pricing.bond import Bond
from ..pricing.curves import ZeroCurve, flat_curve
from ..risk.cs01 import cs01 as _cs01
from ..risk.cs01 import dv01_curve as _dv01_curve


def realized_weekly_vol_bp(series: pd.Series, freq: str = "W-FRI", window_years: float | None = None) -> float:
    """Realized weekly volatility of a yield/spread series, in bp. Input in decimals (0.045 = 4.5%).

    ``window_years``, when given, restricts the calculation to the most recent N years before resampling.
    This matters a lot for a "current risk" report: e.g. the full history of DGS10 (since 1962) includes the
    Volcker-era early-1980s spike, which would badly overstate today's rate volatility if left in. Without a
    window the ENTIRE available history is used, which is appropriate for a long series only if that is what
    you actually want (e.g. for methodological comparisons); for a live risk view, pass a window (e.g. 2-3
    years) so both the rate and the spread volatility describe the SAME, recent regime.
    """
    s = series.dropna()
    if window_years is not None:
        cutoff = s.index.max() - pd.Timedelta(days=window_years * 365.25)
        s = s[s.index >= cutoff]
    if len(s) < 10:
        raise ValueError(f"need at least 10 observations in the window, got {len(s)}")
    weekly = s.resample(freq).last().dropna()
    changes_bp = weekly.diff().dropna() * 1e4
    if len(changes_bp) < 5:
        raise ValueError(f"need at least 5 weekly changes in the window, got {len(changes_bp)}")
    return float(changes_bp.std())


def factor_correlation(rate_series: pd.Series, spread_series: pd.Series, freq: str = "W-FRI",
                       window_years: float | None = None) -> float:
    """Correlation of WEEKLY CHANGES between a rate series and a spread series (both decimals).

    Computed on changes, not levels: level correlation is inflated by shared trends and says little about how
    the two factors co-move week to week, which is what the portfolio-variance formula needs.
    """
    r, s = rate_series.dropna(), spread_series.dropna()
    if window_years is not None:
        cutoff = min(r.index.max(), s.index.max()) - pd.Timedelta(days=window_years * 365.25)
        r, s = r[r.index >= cutoff], s[s.index >= cutoff]
    rw = r.resample(freq).last().diff().dropna()
    sw = s.resample(freq).last().diff().dropna()
    common = rw.index.intersection(sw.index)
    if len(common) < 10:
        raise ValueError(f"need at least 10 overlapping weekly changes, got {len(common)}")
    return float(rw[common].corr(sw[common]))


@dataclass(frozen=True)
class RiskContribution:
    label: str
    notional: float
    dv01: float                    # $ loss per 1bp parallel rate move (curve-consistent: risk.cs01.dv01_curve)
    cs01: float                    # $ loss per 1bp spread move (== dv01 for a parallel shift, by construction)
    rate_vol_bp: float             # realized weekly Treasury yield vol, bp
    spread_vol_bp: float           # realized weekly credit spread vol, bp
    rho: float                     # correlation of weekly rate and spread changes
    rate_risk_contribution: float  # dv01 * rate_vol_bp: MARGINAL piece, ignores correlation
    spread_risk_contribution: float  # cs01 * spread_vol_bp: MARGINAL piece, ignores correlation
    total_risk_1sd: float          # correlation-adjusted total: sqrt(Var(P&L)), see module docstring
    naive_sum_1sd: float           # rate_risk_contribution + spread_risk_contribution, for comparison only
    dominant_factor: str           # "rate" or "spread": whichever MARGINAL contribution is larger


def risk_contribution_report(
    label: str, bond: Bond, settlement: date, curve: ZeroCurve, spread: float, notional: float,
    rate_series: pd.Series, spread_series: pd.Series, window_years: float | None = 3.0,
) -> RiskContribution:
    """Build the DV01/CS01-vs-realized-vol comparison for one bond, including the correlation-adjusted total.

    ``rate_series`` and ``spread_series`` are decimal-yield time series and should be RATING-APPROPRIATE:
    using the same broad-market spread series for every bond regardless of its credit quality makes
    ``dominant_factor`` identical across all bonds by construction (their DV01==CS01, so the split percentage
    reduces to spread_vol / (rate_vol + spread_vol), independent of the bond, if the same two series are
    reused). Pass e.g. ICE AAA/BBB/HY OAS for the matching rating bucket, not one shared proxy.

    ``curve`` should be the REAL (e.g. Treasury) curve, not necessarily flat: DV01 is computed with
    ``risk.cs01.dv01_curve`` (a curve bump), consistent with how CS01 is computed, so both sensitivities are
    correct even when the curve has a realistic slope.

    ``window_years`` (default 3, matching what FRED's ICE OAS series actually cover) restricts both
    volatilities AND the correlation to the same recent window, so the comparison describes TODAY's regime
    rather than a multi-decade average that can include very different periods (e.g. the early-1980s rate
    spike). Pass None to use the full history of each series instead.
    """
    d = _dv01_curve(bond, settlement, curve, spread, notional=notional)
    c = _cs01(bond, settlement, curve, spread, notional=notional)
    rv = realized_weekly_vol_bp(rate_series, window_years=window_years)
    sv = realized_weekly_vol_bp(spread_series, window_years=window_years)
    rho = factor_correlation(rate_series, spread_series, window_years=window_years)
    rate_contrib = d * rv
    spread_contrib = c * sv
    total = float(np.sqrt(max(0.0, d**2 * rv**2 + c**2 * sv**2 + 2 * d * c * rho * rv * sv)))
    return RiskContribution(
        label=label, notional=notional, dv01=d, cs01=c, rate_vol_bp=rv, spread_vol_bp=sv, rho=rho,
        rate_risk_contribution=rate_contrib, spread_risk_contribution=spread_contrib,
        total_risk_1sd=total, naive_sum_1sd=rate_contrib + spread_contrib,
        dominant_factor="spread" if spread_contrib > rate_contrib else "rate",
    )


# Only 3 ICE OAS buckets are available from FRED (AAA, BBB, HY). For ratings outside these three, use the
# nearest bucket's realized volatility as an approximation -- documented here, not hidden in notebook code.
RATING_TO_ICE_BUCKET: dict[str, str] = {
    "AAA": "aaa_oas", "AA": "aaa_oas", "A": "aaa_oas",     # investment-grade, close to AAA behaviour
    "BBB": "bbb_oas",                                       # lowest investment grade, its own bucket
    "BB (HY)": "hy_oas", "B (HY)": "hy_oas", "CCC (HY)": "hy_oas",  # all high-yield -> the HY bucket
}


def spread_series_for_rating(rating: str, series_by_bucket: dict[str, pd.Series]) -> pd.Series:
    """Look up the ICE OAS series (already fetched into ``series_by_bucket``, e.g. {"aaa_oas": df["aaa_oas"],
    "bbb_oas": df["bbb_oas"], "hy_oas": df["hy_oas"]}) for the given rating, via RATING_TO_ICE_BUCKET."""
    if rating not in RATING_TO_ICE_BUCKET:
        raise KeyError(f"unknown rating {rating!r}; expected one of {sorted(RATING_TO_ICE_BUCKET)}")
    bucket = RATING_TO_ICE_BUCKET[rating]
    if bucket not in series_by_bucket:
        raise KeyError(f"rating {rating!r} maps to {bucket!r}, which is missing from series_by_bucket")
    return series_by_bucket[bucket]


def rating_vol_ratio(rating: str, series_by_bucket: dict[str, pd.Series], reference_series: pd.Series,
                     window_years: float | None = 3.0) -> float:
    """How much more (or less) volatile is this rating's spread than the reference (e.g. Baa-30Y proxy)?

    Used to scale a historical widening scenario sized on the long-history proxy down/up to a specific rating
    bucket (see portfolio.scenarios.scaled_historical_scenarios): HY spreads widen far more than AAA spreads in
    a crisis, so applying the SAME bp widening to every bond is unrealistic (see Section H's original bug).
    """
    bucket_series = spread_series_for_rating(rating, series_by_bucket)
    bucket_vol = realized_weekly_vol_bp(bucket_series, window_years=window_years)
    ref_vol = realized_weekly_vol_bp(reference_series, window_years=window_years)
    return bucket_vol / ref_vol


def report_table(reports: list[RiskContribution]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append({
            "bond": r.label, "notional": r.notional, "dv01_$": r.dv01, "cs01_$": r.cs01,
            "rate_vol_bp": r.rate_vol_bp, "spread_vol_bp": r.spread_vol_bp, "rho": r.rho,
            "rate_contribution_$": r.rate_risk_contribution, "spread_contribution_$": r.spread_risk_contribution,
            "naive_sum_1sd_$": r.naive_sum_1sd, "total_risk_1sd_$": r.total_risk_1sd,
            "dominant": r.dominant_factor,
            "spread_share_pct": 100 * r.spread_risk_contribution / (r.rate_risk_contribution + r.spread_risk_contribution),
        })
    return pd.DataFrame(rows).set_index("bond")
