"""Offline tests for the FRED layer (HTTP is mocked; the real network is checked in the Colab notebook)."""
import time
import warnings

import numpy as np
import pandas as pd
import pytest
import requests

from credit_engine.data import fred
from credit_engine.data.fred import (CATALOG, FredError, check_expectations, data_report,
                                     fetch_fred_series, load_series, parse_fred_csv)

NEW_FMT = "observation_date,BAA10Y\n2020-03-16,4.90\n2020-03-17,.\n2020-03-18,5.10\n2020-03-19,\n"
OLD_FMT = "DATE,BAA10Y\n2020-03-18,5.10\n2020-03-16,4.90\n2020-03-16,4.95\n"


class FakeResp:
    def __init__(self, text="", status=200, json_data=None):
        self.text, self.status_code, self._json = text, status, json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Never really wait in retry/backoff tests; record the requested delays."""
    slept = []
    monkeypatch.setattr(fred.time, "sleep", lambda s: slept.append(s))
    return slept


@pytest.fixture
def calls(monkeypatch):
    """Patch requests.get; returns the list of (url, params) actually requested."""
    class Log(list):
        fake = None

    log = Log()

    def fake_get(url, params=None, headers=None, timeout=None):
        log.append((url, params))
        return fake_get.response

    fake_get.response = FakeResp(NEW_FMT)
    monkeypatch.setattr(fred.requests, "get", fake_get)
    log.fake = fake_get
    return log


# ---------------------------------------------------------------- parsing
def test_parse_new_format_handles_dot_and_blank():
    s = parse_fred_csv(NEW_FMT, "BAA10Y")
    assert s.name == "BAA10Y" and len(s) == 4
    assert s.loc["2020-03-16"] == 4.90 and s.loc["2020-03-18"] == 5.10
    assert np.isnan(s.loc["2020-03-17"]) and np.isnan(s.loc["2020-03-19"])


def test_parse_old_format_sorts_and_dedups():
    s = parse_fred_csv(OLD_FMT, "BAA10Y")
    assert list(s.index) == sorted(s.index)
    assert len(s) == 2 and s.loc["2020-03-16"] == 4.95      # last duplicate wins


@pytest.mark.parametrize("bad", ["", "   ", "<html><body>Error</body></html>", "only_one_col\n1\n2\n"])
def test_parse_rejects_garbage(bad):
    with pytest.raises(FredError):
        parse_fred_csv(bad, "X")


# ------------------------------------------------------- download + cache
def test_first_call_downloads_then_cache_hit(tmp_path, calls):
    s1 = fetch_fred_series("baa10y", cache_dir=tmp_path)
    assert len(calls) == 1 and calls[0][0] == fred.FRED_CSV_URL and calls[0][1] == {"id": "BAA10Y"}
    assert (tmp_path / "BAA10Y.csv").exists()
    s2 = fetch_fred_series("BAA10Y", cache_dir=tmp_path)
    assert len(calls) == 1                                   # served from cache
    pd.testing.assert_series_equal(s1, s2)


def test_refresh_and_expired_cache_redownload(tmp_path, calls):
    fetch_fred_series("BAA10Y", cache_dir=tmp_path)
    fetch_fred_series("BAA10Y", cache_dir=tmp_path, refresh=True)
    assert len(calls) == 2
    old = time.time() - 48 * 3600
    import os
    os.utime(tmp_path / "BAA10Y.csv", (old, old))
    fetch_fred_series("BAA10Y", cache_dir=tmp_path)
    assert len(calls) == 3


def test_network_failure_uses_stale_cache_with_warning(tmp_path, calls):
    fetch_fred_series("BAA10Y", cache_dir=tmp_path)

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("no internet")
    fred.requests.get = boom                                 # restored by the monkeypatch fixture
    with pytest.warns(UserWarning, match="stale cache"):
        s = fetch_fred_series("BAA10Y", cache_dir=tmp_path, refresh=True)
    assert len(s) == 4


def test_network_failure_without_cache_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("no internet")
    monkeypatch.setattr(fred.requests, "get", boom)
    with pytest.raises(FredError, match="download failed"):
        fetch_fred_series("BAA10Y", cache_dir=tmp_path)


def test_http_404_gives_series_id_hint(tmp_path, calls):
    calls.fake.response = FakeResp("Not found", status=404)
    with pytest.raises(FredError, match="check the series id"):
        fetch_fred_series("NOPE123", cache_dir=tmp_path)


def test_html_response_not_cached(tmp_path, calls):
    calls.fake.response = FakeResp("<html>captcha</html>")
    with pytest.raises(FredError):
        fetch_fred_series("BAA10Y", cache_dir=tmp_path)
    assert not (tmp_path / "BAA10Y.csv").exists()


def test_api_key_path_and_env_var(tmp_path, calls, monkeypatch):
    calls.fake.response = FakeResp(json_data={"observations": [
        {"date": "2020-03-16", "value": "4.90"}, {"date": "2020-03-17", "value": "."}]})
    s = fetch_fred_series("BAA10Y", cache_dir=tmp_path, api_key="KEY123")
    assert calls[0][0] == fred.FRED_API_URL and calls[0][1]["api_key"] == "KEY123"
    assert s.loc["2020-03-16"] == 4.90 and np.isnan(s.loc["2020-03-17"])
    monkeypatch.setenv("FRED_API_KEY", "ENVKEY")
    fetch_fred_series("BAA10Y", cache_dir=tmp_path, refresh=True)
    assert calls[1][1]["api_key"] == "ENVKEY"


def test_start_end_slicing(tmp_path, calls):
    s = fetch_fred_series("BAA10Y", cache_dir=tmp_path, start="2020-03-17", end="2020-03-18")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2020-03-17", "2020-03-18"]


# ------------------------------------------------- load / report / checks
def _synthetic_csv(series_id, start, end, level=4.0, seed=0):
    idx = pd.bdate_range(start, end)
    vals = level + np.cumsum(np.random.default_rng(seed).normal(0, 0.02, len(idx)))
    return "observation_date," + series_id + "\n" + "\n".join(f"{d.date()},{v:.3f}" for d, v in zip(idx, vals))


def _serve_synthetic(monkeypatch, end="2026-09-17"):
    starts = {"BAA10Y": "1986-01-02", "AAA10Y": "1986-01-02", "DGS10": "1962-01-02",
              "DBAA": "1986-01-02", "DAAA": "1983-01-03", "DGS2": "1976-06-01", "DGS5": "1962-01-02", "DGS20": "1993-10-01", "DGS30": "1977-02-15"}

    def fake_get(url, params=None, headers=None, timeout=None):
        sid = params["id"]
        return FakeResp(_synthetic_csv(sid, starts.get(sid, "2023-09-19"), end))
    monkeypatch.setattr(fred.requests, "get", fake_get)


def test_load_series_converts_percent_to_decimal(tmp_path, calls):
    df = load_series(["baa10y"], cache_dir=tmp_path)
    assert df.loc["2020-03-16", "baa10y"] == pytest.approx(0.049)


def test_load_series_unknown_name():
    with pytest.raises(KeyError, match="Unknown series name"):
        load_series(["not_a_series"])


def test_full_pipeline_with_synthetic_universe(tmp_path, monkeypatch):
    _serve_synthetic(monkeypatch)
    df = load_series(list(CATALOG), cache_dir=tmp_path)
    assert set(df.columns) == set(CATALOG)
    rep = data_report(df)
    assert rep.loc["baa10y", "first"].year == 1986 and rep.loc["hy_oas", "first"].year == 2023
    checks = check_expectations(df, today="2026-09-21")
    assert all(ok for _, ok, _ in checks), [c for c in checks if not c[1]]


def test_check_expectations_flags_problems():
    idx = pd.bdate_range("2015-01-01", "2026-01-01")
    df = pd.DataFrame({"baa10y": 0.02, "hy_oas": 0.04}, index=idx)
    df.loc[df.index < "2023-01-01", "hy_oas"] = np.nan
    df = df.iloc[:400].copy()                       # far too short + stale + starts after 1990
    failed = {n for n, ok, _ in check_expectations(df, today="2026-09-21") if not ok}
    assert any("baa10y: history starts before 1990" in n for n in failed)
    assert any("last observation within 14 days" in n for n in failed)


# ------------------------------------------------- retries and partial loads
def _flaky(monkeypatch, failures, then=None, exc=requests.exceptions.ReadTimeout):
    """requests.get raises `exc` `failures` times, then returns `then`. Returns the call log."""
    log = []

    def fake_get(url, params=None, headers=None, timeout=None):
        log.append(params)
        if len(log) <= failures:
            raise exc("timed out")
        return then if then is not None else FakeResp(NEW_FMT)
    monkeypatch.setattr(fred.requests, "get", fake_get)
    return log


def test_transient_timeouts_are_retried_with_backoff(tmp_path, monkeypatch, no_sleep):
    log = _flaky(monkeypatch, failures=2)
    s = fetch_fred_series("AAA10Y", cache_dir=tmp_path)
    assert len(log) == 3 and len(s) == 4
    assert no_sleep == [2.0, 4.0]                       # exponential backoff


def test_persistent_timeout_raises_after_all_attempts(tmp_path, monkeypatch):
    log = _flaky(monkeypatch, failures=99)
    with pytest.raises(FredError, match=r"after up to 6 attempts.*ReadTimeout"):
        fetch_fred_series("AAA10Y", cache_dir=tmp_path)
    assert len(log) == 6                                # 1 try + 5 retries
    assert not (tmp_path / "AAA10Y.csv").exists()


def test_retries_parameter_respected(tmp_path, monkeypatch):
    log = _flaky(monkeypatch, failures=99)
    with pytest.raises(FredError):
        fetch_fred_series("AAA10Y", cache_dir=tmp_path, retries=0)
    assert len(log) == 1


def test_503_is_retried_but_404_is_not(tmp_path, monkeypatch):
    seq = iter([FakeResp("", 503), FakeResp(NEW_FMT, 200)])
    n = []
    monkeypatch.setattr(fred.requests, "get", lambda *a, **k: (n.append(1), next(seq))[1])
    assert len(fetch_fred_series("BAA10Y", cache_dir=tmp_path)) == 4 and len(n) == 2

    n.clear()
    monkeypatch.setattr(fred.requests, "get", lambda *a, **k: (n.append(1), FakeResp("nf", 404))[1])
    with pytest.raises(FredError, match="check the series id"):
        fetch_fred_series("NOPE", cache_dir=tmp_path)
    assert len(n) == 1                                  # no retry on 404


def _serve_with_one_bad(monkeypatch, bad_id):
    def fake_get(url, params=None, headers=None, timeout=None):
        if params["id"] == bad_id:
            raise requests.exceptions.ReadTimeout("timed out")
        return FakeResp(_synthetic_csv(params["id"], "2020-01-01", "2026-09-17"))
    monkeypatch.setattr(fred.requests, "get", fake_get)


def test_load_series_skip_keeps_good_series_and_records_failure(tmp_path, monkeypatch):
    _serve_with_one_bad(monkeypatch, "AAA10Y")
    with pytest.warns(UserWarning, match="skipped aaa10y"):
        df = load_series(["baa10y", "aaa10y", "dgs10"], cache_dir=tmp_path, on_error="skip")
    assert list(df.columns) == ["baa10y", "dgs10"]
    assert "aaa10y" in df.attrs["failed"]
    checks = check_expectations(df, today="2026-09-21")
    assert ("aaa10y: downloaded", False) in [(n, ok) for n, ok, _ in checks]


def test_load_series_default_raises_on_failure(tmp_path, monkeypatch):
    _serve_with_one_bad(monkeypatch, "AAA10Y")
    with pytest.raises(FredError):
        load_series(["baa10y", "aaa10y"], cache_dir=tmp_path)


def test_rerun_only_downloads_what_is_missing(tmp_path, monkeypatch):
    """The exact Colab scenario: first run times out on one series; second run must not re-download the rest."""
    _serve_with_one_bad(monkeypatch, "AAA10Y")
    with pytest.warns(UserWarning):
        load_series(["baa10y", "aaa10y"], cache_dir=tmp_path, on_error="skip")
    requested = []

    def healthy(url, params=None, headers=None, timeout=None):
        requested.append(params["id"])
        return FakeResp(_synthetic_csv(params["id"], "2020-01-01", "2026-09-17"))
    monkeypatch.setattr(fred.requests, "get", healthy)
    df = load_series(["baa10y", "aaa10y"], cache_dir=tmp_path, on_error="skip")
    assert requested == ["AAA10Y"] and list(df.columns) == ["baa10y", "aaa10y"] and not df.attrs["failed"]


def test_load_series_all_failed_raises(tmp_path, monkeypatch):
    _flaky(monkeypatch, failures=999)
    with pytest.warns(UserWarning, match="skipped baa10y"), \
            pytest.raises(FredError, match="none of the requested series"):
        load_series(["baa10y"], cache_dir=tmp_path, on_error="skip")
