"""FRED download layer.

* Works WITHOUT an API key (public CSV endpoint); an API key is used if given
  (argument or FRED_API_KEY environment variable).
* Every download is cached on disk (default ``data/raw``, git-ignored). Raw
  Moody's / ICE data must not be redistributed, so it is never committed.
* If a refresh fails but an old cache exists, the stale cache is used with a warning.
* The parsing / caching logic is tested offline with mocked HTTP; the real
  network call is verified by running ``notebooks/01_data_check.ipynb``.

Units: FRED spreads and yields are in percent. ``load_series`` converts them to
decimals (4.5 -> 0.045) so they plug directly into the pricing code.
"""
from __future__ import annotations

import io
import os
import time
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
_HEADERS = {"User-Agent": "Mozilla/5.0 (credit-analytics-engine; research use)"}


class FredError(RuntimeError):
    """Raised for download or parsing problems, with a human-readable hint."""


# short name -> (FRED id, unit, description)
CATALOG: dict[str, tuple[str, str, str]] = {
    "baa10y": ("BAA10Y", "percent", "Moody's Baa yield minus 10Y Treasury (daily, from 1986)"),
    "aaa10y": ("AAA10Y", "percent", "Moody's Aaa yield minus 10Y Treasury (daily, from 1986)"),
    "baa_yield": ("DBAA", "percent", "Moody's seasoned Baa corporate bond yield (daily, from 1986)"),
    "aaa_yield": ("DAAA", "percent", "Moody's seasoned Aaa corporate bond yield (daily, from 1983)"),
    "hy_oas": ("BAMLH0A0HYM2", "percent", "ICE BofA US High Yield OAS (FRED keeps ~3y only)"),
    "bbb_oas": ("BAMLC0A4CBBB", "percent", "ICE BofA BBB US Corporate OAS (~3y only)"),
    "aaa_oas": ("BAMLC0A1CAAA", "percent", "ICE BofA AAA US Corporate OAS (~3y only)"),
    "dgs2": ("DGS2", "percent", "2Y Treasury constant maturity yield"),
    "dgs5": ("DGS5", "percent", "5Y Treasury constant maturity yield"),
    "dgs10": ("DGS10", "percent", "10Y Treasury constant maturity yield"),
    "dgs20": ("DGS20", "percent", "20Y Treasury constant maturity yield"),
    "dgs30": ("DGS30", "percent", "30Y Treasury constant maturity yield"),
}


# ------------------------------------------------------------------- parsing
def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse FRED CSV text (header 'observation_date,ID' or 'DATE,ID').

    Missing observations ('.' or empty) become NaN. Result is sorted, unique index.
    """
    head = text.lstrip()[:200].lower()
    if not head or head.startswith("<"):
        raise FredError(f"{series_id}: empty or non-CSV response (HTML error page?)")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # pandas raises several types
        raise FredError(f"{series_id}: could not parse CSV ({exc})") from exc
    if df.shape[1] < 2 or df.empty:
        raise FredError(f"{series_id}: CSV has no data columns/rows")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=series_id)
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index.name = "date"
    return s


# ------------------------------------------------------------------ download
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _get_with_retries(url: str, params: dict, timeout: float, retries: int, backoff: float):
    """GET with exponential backoff on timeouts, connection errors and 429/5xx.

    4xx errors such as 400/404 (bad series id / bad key) are NOT retried.
    """
    for attempt in range(retries + 1):
        last = attempt == retries
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            if r.status_code in _RETRY_STATUS and not last:
                time.sleep(backoff * 2**attempt)
                continue
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if last:
                raise
            time.sleep(backoff * 2**attempt)


def _download_csv_text(series_id: str, api_key: str | None, timeout: float,
                       retries: int = 5, backoff: float = 2.0) -> str:
    try:
        if api_key:
            r = _get_with_retries(FRED_API_URL,
                                  {"series_id": series_id, "api_key": api_key, "file_type": "json"},
                                  timeout, retries, backoff)
            obs = r.json().get("observations")
            if obs is None:
                raise FredError(f"{series_id}: API response has no 'observations'")
            lines = [f"observation_date,{series_id}"]
            lines += [f"{o['date']},{o['value']}" for o in obs]
            return "\n".join(lines)
        r = _get_with_retries(FRED_CSV_URL, {"id": series_id}, timeout, retries, backoff)
        return r.text
    except requests.exceptions.RequestException as exc:
        code = getattr(exc.response, "status_code", None) if getattr(exc, "response", None) is not None else None
        hint = ("check the series id" if code in (400, 404)
                else "FRED may be slow: re-run the cell (finished series are cached), or set FRED_API_KEY")
        raise FredError(f"{series_id}: download failed after up to {retries + 1} attempts "
                        f"(HTTP {code}, {type(exc).__name__}); {hint}") from exc


def fetch_fred_series(
    series_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path = "data/raw",
    max_age_hours: float = 24.0,
    refresh: bool = False,
    api_key: str | None = None,
    timeout: float = 90.0,
    retries: int = 5,
    backoff: float = 2.0,
) -> pd.Series:
    """Return a FRED series (percent units, NaN for missing days), cached on disk."""
    series_id = series_id.upper()
    api_key = api_key or os.environ.get("FRED_API_KEY")
    cache = Path(cache_dir) / f"{series_id}.csv"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600

    if fresh and not refresh:
        text = cache.read_text()
    else:
        try:
            text = _download_csv_text(series_id, api_key, timeout, retries, backoff)
            parse_fred_csv(text, series_id)          # validate BEFORE overwriting the cache
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text)
        except FredError as exc:
            if not cache.exists():
                raise
            warnings.warn(f"{exc} -- using stale cache {cache}", stacklevel=2)
            text = cache.read_text()

    s = parse_fred_csv(text, series_id)
    if start or end:
        s = s.loc[start:end]
    return s


def load_series(names: list[str], *, to_decimal: bool = True, on_error: str = "raise", **kwargs) -> pd.DataFrame:
    """Load several catalog series into one DataFrame (outer join on dates).

    on_error="raise": stop at the first failed download (default, strict).
    on_error="skip" : keep going, warn, and record failures in ``df.attrs["failed"]``.
    Finished downloads are cached, so re-running only retries what is missing.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    cols: dict[str, pd.Series] = {}
    failed: dict[str, str] = {}
    for name in names:
        if name not in CATALOG:
            raise KeyError(f"Unknown series name {name!r}. Known names: {sorted(CATALOG)}")
        series_id, unit, _ = CATALOG[name]
        try:
            s = fetch_fred_series(series_id, **kwargs)
        except FredError as exc:
            if on_error == "raise":
                raise
            failed[name] = str(exc)
            warnings.warn(f"skipped {name}: {exc}", stacklevel=2)
            continue
        cols[name] = s / 100.0 if (to_decimal and unit == "percent") else s
    if not cols:
        raise FredError("none of the requested series could be downloaded: " + "; ".join(failed.values()))
    df = pd.concat(cols, axis=1, sort=True)
    df.index.name = "date"
    df.attrs["failed"] = failed
    return df


# -------------------------------------------------------------------- checks
def data_report(df: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """One row per column: first/last date, #obs, min/max/last (in percent if df is decimal)."""
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            rows.append({"series": c, "first": None, "last": None, "n_obs": 0,
                         "min_pct": None, "max_pct": None, "last_pct": None})
            continue
        rows.append({"series": c, "first": s.index[0].date(), "last": s.index[-1].date(),
                     "n_obs": len(s), "min_pct": s.min() * scale, "max_pct": s.max() * scale,
                     "last_pct": s.iloc[-1] * scale})
    return pd.DataFrame(rows).set_index("series").round(3)


def check_expectations(df: pd.DataFrame, today=None) -> list[tuple[str, bool, str]]:
    """Sanity checks on what we EXPECT from the data. A FAIL is information, not a crash."""
    today = pd.Timestamp(today if today is not None else date.today())
    rep = data_report(df)
    checks: list[tuple[str, bool, str]] = []
    for name, msg in df.attrs.get("failed", {}).items():
        checks.append((f"{name}: downloaded", False, msg))
    for name in ("baa10y", "aaa10y", "dgs10", "baa_yield", "aaa_yield"):
        if name in rep.index and rep.loc[name, "first"] is not None:
            first = pd.Timestamp(rep.loc[name, "first"])
            checks.append((f"{name}: history starts before 1990", first < pd.Timestamp("1990-01-01"),
                           f"first obs {first.date()}"))
    if "hy_oas" in rep.index and rep.loc["hy_oas", "first"] is not None:
        n = int(rep.loc["hy_oas", "n_obs"])
        checks.append(("hy_oas: about 3 years of data (>= 500 obs)", n >= 500,
                       f"{n} obs, first obs {rep.loc['hy_oas', 'first']} (expected ~2023-09 given FRED's rolling window)"))
    for name in rep.index:
        if rep.loc[name, "last"] is None:
            checks.append((f"{name}: has data", False, "no observations"))
            continue
        lag = (today - pd.Timestamp(rep.loc[name, "last"])).days
        checks.append((f"{name}: last observation within 14 days", lag <= 14,
                       f"last obs {rep.loc[name, 'last']} ({lag} days ago)"))
    return checks


def print_checks(checks: list[tuple[str, bool, str]]) -> None:
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  |  {detail}")
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
