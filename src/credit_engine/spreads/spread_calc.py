"""Spread construction, data-gap detection, and validation of proxy spreads.

Why this module exists: FRED only serves ~3 years of ICE OAS, so long-history
work uses Moody's yield minus a Treasury yield as a PROXY. A proxy is only
useful if we measure how close it is to the real thing on the overlap window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def spread(yield_series: pd.Series, benchmark: pd.Series, name: str | None = None) -> pd.Series:
    """yield - benchmark on the dates where both exist (decimals in, decimals out)."""
    df = pd.concat({"y": yield_series, "b": benchmark}, axis=1).dropna()
    out = df["y"] - df["b"]
    out.name = name or "spread"
    return out


def find_gaps(s: pd.Series, min_gap_days: int = 10) -> pd.DataFrame:
    """Calendar gaps between consecutive observations longer than ``min_gap_days``.

    10 days ignores weekends and holidays but catches discontinued series
    (e.g. a Treasury maturity that was not issued for several years).
    """
    idx = pd.DatetimeIndex(s.dropna().index)
    cols = ["last_obs_before", "first_obs_after", "calendar_days"]
    if len(idx) < 2:
        return pd.DataFrame(columns=cols)
    days = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    pos = np.where(days > min_gap_days)[0]
    return pd.DataFrame({cols[0]: idx[pos].date, cols[1]: idx[pos + 1].date, cols[2]: days[pos]})


def gaps_report(df: pd.DataFrame, min_gap_days: int = 10) -> pd.DataFrame:
    """find_gaps for every column; empty frame if no series has a gap."""
    parts = []
    for c in df.columns:
        g = find_gaps(df[c], min_gap_days)
        if not g.empty:
            g.insert(0, "series", c)
            parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["series", "last_obs_before", "first_obs_after", "calendar_days"])
    return pd.concat(parts, ignore_index=True)


def proxy_validation(proxy: pd.Series, reference: pd.Series, freq: str = "W-FRI") -> dict:
    """Compare a long-history proxy spread with a reference OAS on their overlap.

    Reported (spreads in decimals; gaps in bp):
      level_gap_mean_bp / level_gap_std_bp : proxy - reference (a constant gap means a fixed
                                             offset, e.g. maturity/composition difference)
      corr_levels                          : inflated by trends, shown for completeness only
      corr_daily_changes / corr_weekly_changes : do the two move together? (the number that matters)
      beta_weekly                          : reference change per unit of proxy change
                                             (< 1 means the proxy exaggerates moves)
    """
    df = pd.concat({"proxy": proxy, "ref": reference}, axis=1).dropna()
    if len(df) < 60:
        raise ValueError(f"overlap too short ({len(df)} obs); need at least 60")
    gap = df["proxy"] - df["ref"]
    daily = df.diff().dropna()
    weekly = df.resample(freq).last().dropna().diff().dropna()
    if len(weekly) < 20:
        raise ValueError("overlap too short for weekly statistics")
    return {
        "n_days": len(df),
        "overlap_start": df.index[0].date(),
        "overlap_end": df.index[-1].date(),
        "level_gap_mean_bp": gap.mean() * 1e4,
        "level_gap_std_bp": gap.std() * 1e4,
        "corr_levels": df["proxy"].corr(df["ref"]),
        "corr_daily_changes": daily["proxy"].corr(daily["ref"]),
        "corr_weekly_changes": weekly["proxy"].corr(weekly["ref"]),
        "beta_weekly": float(np.polyfit(weekly["proxy"], weekly["ref"], 1)[0]),
    }
