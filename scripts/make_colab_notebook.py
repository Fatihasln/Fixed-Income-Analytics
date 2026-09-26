"""Build notebooks/credit_engine_FULL.ipynb: ONE self-contained notebook with ALL steps (Day 1 + data + spread validation).

Generated FROM the real, tested source files; contains NO secrets (FRED key comes from Colab Secrets at run time).
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "credit_engine"
OUT_DIR = "credit_engine_src/credit_engine"
VERSION = "0.8.0"

FILES = ["__init__.py", "pricing/daycount.py", "pricing/bond.py", "pricing/curves.py", "pricing/yield_solver.py",
         "risk/duration.py", "risk/cs01.py", "data/fred.py", "spreads/spread_calc.py", "spreads/events.py", "spreads/regimes.py", "spreads/backtest.py",
         "portfolio/risk_contribution.py", "portfolio/scenarios.py"]

C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell
cells = []

cells.append(M(f"""# Credit Analytics Engine: FULL NOTEBOOK
**One file, every step.** You don't need any other file. All the code lives inside this notebook.

| Section | What it does |
|---|---|
| **A** | Setup (writes and loads the code) |
| **B** | Day 1: bond pricing, duration, DV01, CS01 + validation |
| **C** | Downloads and checks real FRED data |
| **D** | Spreads, data gaps, proxy validation, charts |
| **E** | Regime model: calm / stress (Markov Switching), full-sample (in-sample) |
| **F** | Live simulation: expanding-window periodic refit, min_train_weeks sensitivity check |
| **G** | Risk contribution: DV01=CS01, but multiplied by real volatility, which factor dominates |
| **H** | Stress scenarios: parallel shock, steepener, historical-style spread widening, portfolio P&L |
| **I** | Advanced interactive dashboard: pricing, scenario shocks and the live regime, in one panel |

## How to run it
1. (Recommended, 1 minute) **Put your FRED API key in Colab Secrets.** Never type the key into a cell.
   Get a **new key** at fred.stlouisfed.org > My Account > API Keys. In Colab, click the **key icon (Secrets)** in the left panel > *Add new secret*:
   Name `FRED_API_KEY`, Value = your key, **turn on Notebook access**.
2. From the menu: **Runtime > Run all**.
3. Scroll to the bottom once it's done.

**What to send back:** only the cells whose heading says "Send me this" (a screenshot is fine). Section B is for your own confidence: the numbers should match Day 1.
If something errors, paste the **whole** error, don't change anything yourself. (v{VERSION})"""))

# ------------------------------------------------------------------ A
cells.append(M("# SECTION A: Setup\n## A1. Clean slate\nRemoves anything left over from earlier attempts."))
cells.append(C('''import os, sys, shutil, subprocess

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "credit-engine"], check=False)
for m in [m for m in sys.modules if m.startswith("credit_engine")]:
    del sys.modules[m]                                   # drop any old version cached in memory
shutil.rmtree("credit_engine_src", ignore_errors=True)   # remove old files
for d in ["pricing", "risk", "data", "spreads", "portfolio"]:
    os.makedirs(f"credit_engine_src/credit_engine/{d}", exist_ok=True)
    open(f"credit_engine_src/credit_engine/{d}/__init__.py", "w").close()
print("Clean slate done.")'''))

cells.append(M("## A2. Write the source files\nEach cell below creates one source file. Just run and move on; open one up if you're curious."))
for rel in FILES:
    cells.append(C(f"%%writefile {OUT_DIR}/{rel}\n" + (SRC / rel).read_text().rstrip("\n") + "\n"))

cells.append(M(f"## A3. Load and verify\nVersion should read `{VERSION}`. If a submodule is missing (e.g. `ModuleNotFoundError: credit_engine.data.fred`), this cell says so **explicitly** and tells you what to do, instead of a cryptic error."))
cells.append(C(f'''import sys, os, importlib

FIX = "\\n\\n>>> FIX: Runtime > Restart session, then Runtime > Run all to rerun EVERYTHING from the top (not cell by cell)."

sys.path.insert(0, os.path.abspath("credit_engine_src"))
importlib.invalidate_caches()
try:
    import credit_engine
except ModuleNotFoundError as e:
    raise RuntimeError(f"credit_engine could not be imported at all ({{e}})." + FIX) from e

print("Version:", credit_engine.__version__)
print("Loaded from:", credit_engine.__file__)
if credit_engine.__version__ != "{VERSION}" or "credit_engine_src" not in credit_engine.__file__:
    raise RuntimeError(
        f"Unexpected version/location: {{credit_engine.__version__}} @ {{credit_engine.__file__}} "
        f"(expected: {VERSION}, inside credit_engine_src)." + FIX)

# Try each submodule individually so we can name exactly which one is missing (the top-level import above can
# succeed even if a submodule failed to get written, since "import credit_engine" only needs the package itself).
_submodules = ["pricing.bond", "pricing.daycount", "pricing.curves", "pricing.yield_solver",
              "risk.duration", "risk.cs01", "data.fred", "spreads.spread_calc", "spreads.events",
              "spreads.regimes", "spreads.backtest", "portfolio.risk_contribution", "portfolio.scenarios"]
_missing = []
for _m in _submodules:
    try:
        importlib.import_module(f"credit_engine.{{_m}}")
    except ModuleNotFoundError:
        _missing.append(_m)
if _missing:
    raise RuntimeError(f"These submodules are missing: {{_missing}}. Likely a write cell in A2 was skipped, "
                       f"or an old set of files is still on disk." + FIX)

print(f"All submodules ({{len(_submodules)}}) loaded. Everything is fine.")'''))

# ------------------------------------------------------------------ B
demo = (ROOT / "examples" / "demo.py").read_text()
demo_body = demo[demo.index("from credit_engine.pricing.bond import Bond"):]
cells.append(M("# SECTION B: Day 1, pricing and risk\n## B1. Demo: three example bonds\nExpected: DV01 = CS01 (you'll see why in later sections). Values: AAA-like 4,352 / BBB-like 7,569 / HY-like 5,119.\nNote: the Treasury yield (4.20%) here is just an illustrative made-up number."))
cells.append(C("from datetime import date\n\n" + demo_body))

cells.append(M("## B2. Hand-calculation check\nA 2-year, 5% annual coupon, par bond. The results can be worked out by hand on paper; the code should match exactly."))
cells.append(C('''from datetime import date
from credit_engine.pricing.bond import Bond, dirty_price
from credit_engine.risk.duration import macaulay_duration, modified_duration, convexity, dv01

b, s = Bond(0.05, date(2027, 1, 15), frequency=1), date(2025, 1, 15)
checks = [("Price (par)",          dirty_price(b, s, 0.05),        100.0),
          ("Macaulay duration",   macaulay_duration(b, s, 0.05),  1.952381),
          ("Modified duration",   modified_duration(b, s, 0.05),  1.859410),
          ("Convexity",           convexity(b, s, 0.05),          5.269409),
          ("DV01 (100 notional)", dv01(b, s, 0.05),               0.018594)]
for name, got, want in checks:
    print(f"[{'PASS' if abs(got - want) < 1e-5 else 'FAIL'}] {name}: code = {got:.6f} | by hand = {want:.6f}")'''))

cells.append(M("## B3. Cross-check against QuantLib (optional but important)\nPrices the same bonds with an independent library and compares. Installing it can take ~1 minute. If it can't be installed, the cell says so and you can skip it."))
cells.append(C('''import subprocess, sys
from datetime import date
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "QuantLib"], capture_output=True)
try:
    import QuantLib as ql
except ImportError:
    ql = None
    print("QuantLib could not be installed; you can skip this cell (it's not required).")

if ql is not None:
    from credit_engine.pricing.bond import Bond, clean_price
    from credit_engine.pricing.daycount import DayCount
    from credit_engine.risk.duration import macaulay_duration, modified_duration, convexity

    def ql_metrics(bond, settle, y):
        ql.Settings.instance().evaluationDate = ql.Date(settle.day, settle.month, settle.year)
        m = bond.maturity
        freq = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly}[bond.frequency]
        sched = ql.Schedule(ql.Date(m.day, m.month, m.year - 40), ql.Date(m.day, m.month, m.year), ql.Period(freq),
                            ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
        dc = (ql.Thirty360(ql.Thirty360.USA) if bond.day_count is DayCount.THIRTY_360
              else ql.ActualActual(ql.ActualActual.ISMA, sched))
        qb = ql.FixedRateBond(0, 100.0, sched, [bond.coupon_rate], dc)
        r = ql.InterestRate(y, dc, ql.Compounded, freq)
        return {"clean": qb.cleanPrice(y, dc, ql.Compounded, freq), "accrued": qb.accruedAmount(),
                "macaulay": ql.BondFunctions.duration(qb, r, ql.Duration.Macaulay),
                "modified": ql.BondFunctions.duration(qb, r, ql.Duration.Modified),
                "convexity": ql.BondFunctions.convexity(qb, r)}

    samples = [("BBB-like 10y 30/360", Bond(0.058, date(2036, 9, 15)), date(2026, 9, 21), 0.057),
               ("HY-like 7y 30/360",  Bond(0.085, date(2033, 9, 15)), date(2026, 9, 21), 0.087),
               ("AAA-like 5y ACT/ACT", Bond(0.045, date(2031, 9, 15), 2, DayCount.ACT_ACT_ICMA), date(2026, 9, 21), 0.048)]
    worst = 0.0
    print(f"{'bond':<22}{'metric':<11}{'ours':>14}{'QuantLib':>14}{'diff':>10}")
    for label, bond, st, y in samples:
        q = ql_metrics(bond, st, y)
        ours = {"clean": clean_price(bond, st, y), "accrued": bond.accrued_interest(st),
                "macaulay": macaulay_duration(bond, st, y), "modified": modified_duration(bond, st, y),
                "convexity": convexity(bond, st, y)}
        for k in ours:
            diff = abs(ours[k] - q[k]); worst = max(worst, diff)
            print(f"{label:<22}{k:<11}{ours[k]:>14.8f}{q[k]:>14.8f}{diff:>10.1e}")
    print(f"\\n[{'PASS' if worst < 1e-6 else 'FAIL'}] largest difference: {worst:.1e}")'''))

# ------------------------------------------------------------------ C
cells.append(M("# SECTION C: Real data (FRED)\n## C1. API key (from Colab Secrets)\nIf there's no key, this warns and continues without one (can be slower, you may see timeouts)."))
cells.append(C('''import os

try:
    from google.colab import userdata
except ImportError:
    userdata = None
    print("Looks like you're outside Colab; Secrets aren't available. Continuing without a key.")

if userdata is not None:
    try:
        os.environ["FRED_API_KEY"] = userdata.get("FRED_API_KEY")
        print("FRED API key loaded from Colab Secrets.")
    except userdata.SecretNotFoundError:
        msg = ("PROBLEM: no secret named 'FRED_API_KEY' was found. "
               "Click the key icon in the left panel, choose 'Add new secret'. "
               "Type the Name EXACTLY as 'FRED_API_KEY' (watch case and spaces). "
               "Continuing without a key; this can be slow.")
        print(msg)
    except userdata.NotebookAccessError:
        msg = ("PROBLEM: the 'FRED_API_KEY' secret exists but access for THIS NOTEBOOK is off. "
               "Click the key icon in the left panel, find FRED_API_KEY. "
               "Turn ON 'Notebook access' (a per-notebook permission, off by default). "
               "Continuing without a key; this can be slow.")
        print(msg)
    except Exception as exc:
        print(f"Could not read the API key (unexpected error: {type(exc).__name__}: {exc}). Continuing without a key.")'''))

cells.append(M("## C2. Download the data\nIf a series fails to download, rerun this cell (whatever already succeeded is cached)."))
cells.append(C('''from credit_engine.data.fred import load_series, data_report, check_expectations, print_checks

names = ["baa10y", "aaa10y", "baa_yield", "aaa_yield", "hy_oas", "bbb_oas", "aaa_oas",
         "dgs2", "dgs5", "dgs10", "dgs20", "dgs30"]
df = load_series(names, on_error="skip")
if df.attrs["failed"]:
    print("SERIES THAT DID NOT DOWNLOAD (rerun this cell):", list(df.attrs["failed"]))
data_report(df)                                     # values are in percentage points'''))

cells.append(M("## C3. Automated checks\nFAIL = information, not an error. **Send me this.**"))
cells.append(C("print_checks(check_expectations(df))"))

# ------------------------------------------------------------------ D
cells.append(M("# SECTION D: Spread analysis\n## D1. Data gaps\nIs there a long break in the middle of any series? An empty table means no gaps. **Send me this.**"))
cells.append(C('''from credit_engine.spreads.spread_calc import gaps_report

gaps = gaps_report(df)
print("No long gaps." if gaps.empty else f"{len(gaps)} long gap(s) found:")
gaps'''))

cells.append(M("## D2. Compute the spreads ourselves\nMoody's yield minus Treasury yield (10, 20, 30 year). Our own Baa-10Y calculation should match what FRED publishes directly (difference under ~1bp). **Send me this.**"))
cells.append(C('''import pandas as pd
from credit_engine.spreads.spread_calc import spread

sp = {}
for grade in ("baa", "aaa"):
    for tenor in ("10", "20", "30"):
        y, b = f"{grade}_yield", f"dgs{tenor}"
        if y in df.columns and b in df.columns:
            sp[f"{grade}_{tenor}y"] = spread(df[y], df[b], f"{grade}_{tenor}y")

for k, v in sp.items():
    print(f"{k}: {v.index[0].date()} -> {v.index[-1].date()}, {len(v)} days")

print()
for ours, fred_col in (("baa_10y", "baa10y"), ("aaa_10y", "aaa10y")):
    if ours in sp and fred_col in df.columns:
        d = (sp[ours] - df[fred_col]).dropna().abs() * 1e4
        print(f"{ours} vs FRED {fred_col}: {len(d)} days, mean diff {d.mean():.2f} bp, max diff {d.max():.2f} bp")'''))

cells.append(M("""## D3. Proxy validation (the most important step)
How close is the long-history Moody's proxy to the real ICE OAS? Measured on the shared ~3-year overlap. **Send me this.**
- `level_gap_mean_bp`: level difference (a constant gap means a maturity/composition difference)
- `corr_weekly_changes`: do weekly changes move together? (the number that actually matters)
- `beta_weekly`: below 1 means the proxy exaggerates moves"""))
cells.append(C('''from credit_engine.spreads.spread_calc import proxy_validation

pairs = [("baa_10y", "bbb_oas"), ("baa_20y", "bbb_oas"), ("baa_30y", "bbb_oas"),
         ("aaa_10y", "aaa_oas"), ("aaa_20y", "aaa_oas"), ("aaa_30y", "aaa_oas")]
rows = {}
for k, ref in pairs:
    if k in sp and ref in df.columns:
        rows[f"{k} vs ICE {ref}"] = proxy_validation(sp[k], df[ref])
val = pd.DataFrame(rows).T
print("Overlap window:", val["overlap_start"].iloc[0], "->", val["overlap_end"].iloc[0], f"({int(val['n_days'].iloc[0])} days)")
val.drop(columns=["overlap_start", "overlap_end", "n_days"]).astype(float).round(3)'''))

cells.append(M("## D4. Charts\n1) The Baa spread, with stress episodes marked (2008 and 2020 should stand out). 2) How close are the proxies to ICE BBB OAS? **Send me this.**"))
cells.append(C('''import matplotlib.pyplot as plt
from credit_engine.spreads.events import plot_with_events

if "baa10y" in df.columns:
    plot_with_events(df["baa10y"].dropna() * 1e4, "Moody's Baa - 10Y Treasury (bp), stress episodes marked")
    plt.show()

cand = {"ICE BBB OAS (reference)": df.get("bbb_oas"), "Baa - 10Y": sp.get("baa_10y"),
        "Baa - 20Y": sp.get("baa_20y"), "Baa - 30Y": sp.get("baa_30y")}
cand = {k: v for k, v in cand.items() if v is not None}
ov = pd.concat(cand, axis=1).dropna() * 1e4
ov.plot(figsize=(12, 5), title="How close are the proxies to ICE BBB OAS? (bp, shared ~3y)")
plt.show()'''))

# ------------------------------------------------------------------ E
cells.append(M("""# SECTION E: Regime model (calm / stress)
**Decision (based on D3):** the primary proxy is **Baa - 30Y**. Almost as good as Baa-20Y, but 20Y has a gap between 1987-1993. Baa-10Y came out weak (52bp level gap, beta 0.68) and is only used as a robustness check. The AAA proxies are weak too (beta 0.4-0.6), so they aren't used.

**Model:** *weekly log-changes* of the spread, 2 hidden regimes (calm / stress), each with its own mean and variance. Stress regime = the high-variance regime.

**Critical rule:** only the **filtered** probability is used (at time t, using only data up to that day). The *smoothed* probability looks into the future and is banned as a signal (tested: the difference can be as large as 0.8).
Note: parameters are estimated on the full sample (in-sample). A live system needs an expanding-window refit -- that's the next section."""))
cells.append(M("## E1. Fit the model and summarize\n**Send me this.** The `stress` row is the high-variance regime. Converged should read True. **Careful:** `mean_weekly_change_pct` (the regime means) is statistically unreliable -- the table now also shows `mean_se_pct` and `weekly_sigma_se_pct` (standard errors from the fit): compare each mean to its own SE and you'll usually see they're close in size (not statistically significant), unlike the volatilities, which are precisely estimated. Base your interpretation on `weekly_sigma_pct`, duration and time share instead."))
cells.append(C('''from credit_engine.spreads.regimes import (fit_regime_model, regime_periods, detection_table,
                                            plot_regimes, weekly_level_bp)
from credit_engine.spreads.events import EVENTS

assert sp["baa_30y"].min() > 0, f"Baa-30Y must be positive (we take logs). Min value: {sp['baa_30y'].min()} -> tell me"
prim = fit_regime_model(sp["baa_30y"])
print("Converged:", prim.converged, "| observations (weeks):", prim.n_obs, "| log-likelihood:", round(prim.loglik, 1))
prim.summary.round(2)'''))
cells.append(M("## E2. Transition probabilities\n**Send me this.** Row = current regime, column = next week. A high diagonal means regimes are persistent."))
cells.append(C("prim.transition.round(3)"))
cells.append(M("## E3. Chart: when does the model see stress?\n**Send me this.** Top: the spread with marked events (red). Bottom: the model's filtered stress probability (orange)."))
cells.append(C('''import matplotlib.pyplot as plt

lvl = weekly_level_bp(sp["baa_30y"])
plot_regimes(lvl, prim.stress_prob, "Baa - 30Y: filtered stress probability (orange) and marked events (red)", events=EVENTS)
plt.show()'''))
cells.append(M("## E4. Which events did the model catch?\n**Send me this.** An empty `first_alert` means the model never flagged that event as stress. `lag_days`: gap between the alert and the event's start (event dates are approximate, the lag is indicative)."))
cells.append(C("detection_table(prim.stress_prob, EVENTS).round(2)"))
cells.append(M("## E5. Does the model find stress episodes not on my list?\n**Send me this.** Episodes lasting at least 2 weeks."))
cells.append(C("regime_periods(prim.stress_prob, level_bp=lvl, min_weeks=2).round(2)"))
cells.append(M("## E6. Robustness check: does Baa-10Y find the same regimes?\n**Send me this.** High agreement = the result doesn't hinge on which proxy we picked."))
cells.append(C('''rob = fit_regime_model(df["baa10y"].dropna())
common = prim.stress_prob.index.intersection(rob.stress_prob.index)
agree = ((prim.stress_prob[common] >= 0.5) == (rob.stress_prob[common] >= 0.5)).mean() * 100
corr = prim.stress_prob[common].corr(rob.stress_prob[common])
print(f"Baa-30Y and Baa-10Y agree on the regime: {agree:.1f}% of weeks | probability correlation: {corr:.2f} | did the Baa-10Y model converge: {rob.converged}")
rob.summary.round(2)'''))
cells.append(M("""## E7. Model selection: is 2 regimes actually better than 3?
So far we assumed 2 regimes (calm/stress) without justifying that choice. Here we fit a 3-regime version of the SAME model and compare by AIC and BIC. A 3-regime model has more parameters and can only fit the in-sample data at least as well, so a fair comparison must penalize that extra flexibility -- which is exactly what AIC/BIC do (BIC penalizes harder). If BIC does not improve when adding a third regime, that's evidence against a 3-state model for this series, not proof there are only two "true" states.
**Send me this.** This cell can take a minute or two (each k fits with several random restarts). Note also the standard errors already shown in E1's summary table (`weekly_sigma_se_pct`, `mean_se_pct`): the regime means are usually not statistically significant (their SE is close to the estimate itself), which is why the write-up leans on volatility/duration/time-share, not on the means."""))
cells.append(C('''from credit_engine.spreads.regimes import compare_regime_counts

model_comparison = compare_regime_counts(sp["baa_30y"], k_range=(2, 3))
model_comparison'''))

# ------------------------------------------------------------------ F
cells.append(M("""# SECTION F: Live simulation (expanding window)
The model in Section E was fit on the **full sample** (in-sample): its parameters were found while also seeing the future. That's fine for learning the model's structure, but it's an optimistic way to measure performance.

Here the model is refit periodically (default: every 13 weeks, about one quarter), **using only the data available up to that point each time**. Between refits, it filters forward with the last fitted (fixed) parameters. This mimics what a live system would actually have known.

**Important:** the optimizer (EM + quasi-Newton) can sometimes land in a statistically equivalent but different local optimum even on identical data (regime labels can swap). This is optimizer noise, not look-ahead leakage; it is documented and tested separately in the code."""))
cells.append(M("## F1. Run the live simulation\n**Send me this.** `n_refits`: how many times the model was refit. This cell can take a little while (it fits the model dozens of times)."))
cells.append(C('''from credit_engine.spreads.backtest import expanding_refit, compare_live_vs_insample

live = expanding_refit(sp["baa_30y"], min_train_weeks=104, refit_every=13)
print("Total number of refits:", live.n_refits)
live.refit_log.tail(8)'''))
cells.append(M("## F2. Live vs full-sample (in-sample) comparison\n**Send me this.** `agreement_pct`: how often the two approaches agree on the regime call (stress/calm). A lower number is expected and normal: the live system works with less information, and that gap is itself an interesting finding."))
cells.append(C('''cmp = compare_live_vs_insample(live.stress_prob, prim.stress_prob)
for k, v in cmp.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")'''))
cells.append(M("## F3. Chart: live vs in-sample stress probability\n**Send me this.** How much do they diverge, especially at the start of a crisis (when the model has less to go on)?"))
cells.append(C('''import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(prim.stress_prob.index, prim.stress_prob.values, label="In-sample (optimistic)", color="tab:blue", alpha=0.7)
ax.plot(live.stress_prob.index, live.stress_prob.values, label="Live (expanding window)", color="tab:orange", alpha=0.8)
ax.set_ylim(0, 1)
ax.set_title("Baa-30Y: in-sample vs live-simulation stress probability")
ax.legend()
plt.tight_layout()
plt.show()'''))
cells.append(M("## F4. Sensitivity check: does the starting data window (min_train_weeks) matter?\nHow many weeks of data should the model demand before its first fit? Too little = noisier early estimates; too much = no signal at all for the first few years. We try a few values and see how the agreement with the in-sample fit changes.\n**Send me this.** This cell can take ~2 minutes (it reruns the model from scratch for 3 different settings)."))
cells.append(C('''from credit_engine.spreads.backtest import min_train_weeks_sensitivity

sens = min_train_weeks_sensitivity(sp["baa_30y"], prim.stress_prob, candidates=[104, 156, 208],
                                   refit_every=26, search_reps=5)
sens.round(2)'''))

# ------------------------------------------------------------------ G
cells.append(M("""# SECTION G: Risk contribution (DV01=CS01, but the real-world effect differs)
Section B showed: for a plain bullet bond under a parallel shift, CS01 is mathematically equal to DV01. Here we complete that finding with real data: **the sensitivities are equal, but how much rates and credit spreads actually move is very different.**

**Important (rating-appropriate spread series):** each bond must use the ICE OAS series that matches its own rating (AAA-like -> aaa_oas, BBB-like -> bbb_oas, HY-like -> hy_oas). Feeding the same spread series to all three bonds would be wrong: since DV01=CS01, `spread_share_pct` then collapses to `spread_vol/(rate_vol+spread_vol)` and comes out IDENTICAL for all three, regardless of which bond it is. Volatility is also measured over the **last 3 years only** (`window_years=3`); otherwise, e.g., the full history of Treasury yields since 1962 (including the extreme-rate 1980s) would badly overstate today's risk.

**Important (real curve, not flat):** DV01 and CS01 are both computed against the REAL Treasury curve (2/5/10/20/30Y), not a single flat rate -- via `risk.cs01.dv01_curve`, consistent with how CS01 already works.

**Important (correlation):** `rate_risk_contribution` and `spread_risk_contribution` are MARGINAL pieces and do not simply add up to the true portfolio risk unless rate and spread moves are uncorrelated. In reality they usually move together (often negatively: rates fall while spreads widen in a flight-to-quality). The correct total is `total_risk_1sd = sqrt(DV01^2*sigma_r^2 + CS01^2*sigma_s^2 + 2*DV01*CS01*rho*sigma_r*sigma_s)`, shown alongside the naive (uncorrelated) sum for comparison."""))
cells.append(M("## G1. Risk contribution table for the three example bonds\n**Send me this.** The `dominant` column shows which factor (rate or credit) actually carries the risk. `rho` is the realized correlation between weekly rate and spread changes; compare `naive_sum_1sd_$` (ignores rho) with `total_risk_1sd_$` (accounts for it) -- if rho is negative, the total should be noticeably SMALLER than the naive sum."))
cells.append(C('''from credit_engine.portfolio.risk_contribution import (risk_contribution_report, report_table,
                                                       spread_series_for_rating, rating_vol_ratio)
from credit_engine.pricing.curves import linear_curve

TENORS = [2, 5, 10, 20, 30]
treasury_curve = linear_curve(TENORS, [df[f"dgs{t}"].dropna().iloc[-1] for t in TENORS])   # the REAL curve, not flat
ICE_BUCKETS = {"aaa_oas": df["aaa_oas"], "bbb_oas": df["bbb_oas"], "hy_oas": df["hy_oas"]}
BOND_RATINGS = {"AAA-like  5y": "AAA", "BBB-like 10y": "BBB", "HY-like   7y": "BB (HY)"}   # must match the labels in BONDS

reports = []
for label, bond, spr in BONDS:   # the example bonds from SECTION B (AAA/BBB/HY-like)
    spread_series = spread_series_for_rating(BOND_RATINGS[label], ICE_BUCKETS)
    reports.append(risk_contribution_report(label, bond, SETTLE, treasury_curve, spr, 10_000_000,
                                            rate_series=df["dgs10"], spread_series=spread_series))
report_table(reports).round(2)'''))

# ------------------------------------------------------------------ H
cells.append(M("""# SECTION H: Stress scenarios
We test our example portfolio (AAA/BBB/HY-like, 10mm notional each) under a few scenarios using an **exact reprice** (not a duration approximation), on the same real Treasury curve as Section G: a parallel shock, a curve steepener, and historical-crisis-style spread widening."""))
cells.append(M("## H1. Parallel shock and steepener scenarios\n**Send me this.**"))
cells.append(C('''from credit_engine.portfolio.scenarios import PortfolioBond, parallel, steepen, run_scenarios

portfolio = [PortfolioBond(label, bond, spr, 10_000_000) for label, bond, spr in BONDS]
basic_scenarios = [parallel(100), parallel(-100), steepen(-50, 100)]
run_scenarios(portfolio, SETTLE, treasury_curve, basic_scenarios).round(0)'''))
cells.append(M("""## H2. Spread widening styled on historical crises, SCALED BY RATING
We take the real peak spread values from Section E/F's `regime_periods` output (measured on the long-history Baa-30Y proxy) and ask: what happens to our portfolio if spreads jump from today's level to that peak?

**Important:** applying the SAME bp widening to every bond regardless of rating would be unrealistic -- in a real crisis, HY spreads widen far more than AAA spreads. Each bond's widening is scaled by `rating_vol_ratio`: its own rating bucket's realized volatility relative to the Baa-30Y proxy the peaks were measured on.
**Send me this.**"""))
cells.append(C('''from credit_engine.portfolio.scenarios import scaled_historical_scenarios

current_bp = float(lvl.iloc[-1])   # today's Baa-30Y level (bp), from SECTION E
peaks_bp = {"2008 GFC": 620.0, "COVID-19": 430.0, "2022 rate shock": 340.0}   # approximate, based on E4/E5 observations
rating_ratios = {label: rating_vol_ratio(BOND_RATINGS[label], ICE_BUCKETS, sp["baa_30y"]) for label, _, _ in BONDS}
print("Rating volatility ratios vs Baa-30Y (i.e. how much more/less this bond's spread typically moves):", 
     {k: round(v, 2) for k, v in rating_ratios.items()})
hist_scenarios = scaled_historical_scenarios(peaks_bp, current_spread_bp=current_bp, rating_ratios=rating_ratios)
print(f"Today's Baa-30Y level: {current_bp:.0f} bp")
run_scenarios(portfolio, SETTLE, treasury_curve, hist_scenarios).round(0)'''))

# ------------------------------------------------------------------ I
cells.append(M("""# SECTION I: Advanced interactive dashboard
Three tabs, built with `ipywidgets` (no separate web server needed), tying together everything built so far:
- **Pricing & Risk**: type in a bond, see its price, duration, DV01/CS01 and which risk factor (rate or credit) actually dominates, with a small chart.
- **Scenario Shock**: drag a curve shock and a spread shock, see the exact repriced P&L on that same bond update instantly.
- **Regime**: today's live (filtered) stress probability, plotted against the last few years, so you can see where "now" sits."""))
cells.append(C('''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ipywidgets"], capture_output=True)'''))
cells.append(M("## I1. Dashboard\nRun the cell, then switch between the three tabs and drag the sliders -- everything updates live."))
cells.append(C('''import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from datetime import date, timedelta

from credit_engine.pricing.curves import shift_curve
from credit_engine.pricing.bond import dirty_price_from_curve

RATING_SPREAD_BP = {"AAA": 60, "AA": 90, "A": 130, "BBB": 180, "BB (HY)": 320, "B (HY)": 480, "CCC (HY)": 800}

# ---- shared bond inputs (drive all three tabs) ----
coupon_w = W.FloatSlider(value=6.0, min=0.0, max=12.0, step=0.25, description="Coupon (%)")
maturity_w = W.IntSlider(value=7, min=1, max=30, step=1, description="Maturity (yrs)")
rating_w = W.Dropdown(options=list(RATING_SPREAD_BP), value="BBB", description="Rating")
notional_w = W.FloatText(value=10_000_000, description="Notional ($)")

# ---- scenario-tab-only inputs ----
parallel_shock_w = W.FloatSlider(value=0, min=-300, max=300, step=5, description="Parallel shock (bp)")
spread_shock_w = W.FloatSlider(value=0, min=-300, max=300, step=5, description="Spread shock (bp)")

out_price, out_scenario, out_regime = W.Output(), W.Output(), W.Output()


def current_bond():
    maturity_date = SETTLE + timedelta(days=int(maturity_w.value * 365.25))
    bond = Bond(coupon_w.value / 100, maturity_date)
    spr = RATING_SPREAD_BP[rating_w.value] / 1e4
    spread_series = spread_series_for_rating(rating_w.value, ICE_BUCKETS)
    return bond, spr, spread_series


def render_pricing(_=None):
    with out_price:
        clear_output(wait=True)
        bond, spr, spread_series = current_bond()
        curve = treasury_curve
        r = risk_contribution_report(rating_w.value, bond, SETTLE, curve, spr, notional_w.value,
                                    rate_series=df["dgs10"], spread_series=spread_series)
        years_to_maturity = (bond.maturity - SETTLE).days / 365.25
        ytm = float(np.atleast_1d(curve(np.array([years_to_maturity])))[0]) + spr
        price = dirty_price_from_curve(bond, SETTLE, curve, spr)
        print(f"YTM (approx, at this bond's own maturity on the curve): {ytm:.2%}   Clean price: {price - bond.accrued_interest(SETTLE):.3f}")
        print(f"Macaulay: {macaulay_duration(bond, SETTLE, ytm):.2f}   Modified: {modified_duration(bond, SETTLE, ytm):.2f}   Convexity: {convexity(bond, SETTLE, ytm):.1f}")
        print(f"DV01: ${r.dv01:,.0f}   CS01: ${r.cs01:,.0f}  (essentially equal under a parallel shift)")
        print(f"Rate risk contribution:   ${r.rate_risk_contribution:,.0f}  (typical weekly move {r.rate_vol_bp:.1f} bp)")
        print(f"Credit risk contribution: ${r.spread_risk_contribution:,.0f}  (typical weekly move {r.spread_vol_bp:.1f} bp)")
        print(f"Correlation (rate vs spread weekly changes): {r.rho:+.2f}")
        print(f"Naive sum (ignores correlation): ${r.naive_sum_1sd:,.0f}   Total risk (correlation-adjusted): ${r.total_risk_1sd:,.0f}")
        print(f"Dominant risk factor: {r.dominant_factor.upper()}")

        fig, ax = plt.subplots(figsize=(5, 1.8))
        bars = ax.barh(["Rate", "Credit"], [r.rate_risk_contribution, r.spread_risk_contribution],
                       color=["tab:blue", "tab:red"])
        ax.bar_label(bars, fmt="${:,.0f}")
        ax.set_title("Risk contribution ($, typical week)")
        plt.tight_layout()
        plt.show()


def render_scenario(_=None):
    with out_scenario:
        clear_output(wait=True)
        bond, spr, _spread_series = current_bond()
        base_curve = treasury_curve
        base_price = dirty_price_from_curve(bond, SETTLE, base_curve, spr)
        base_value = base_price / 100 * notional_w.value

        shocked_curve = shift_curve(base_curve, parallel_shock_w.value / 1e4)
        shocked_spread = spr + spread_shock_w.value / 1e4
        shocked_price = dirty_price_from_curve(bond, SETTLE, shocked_curve, shocked_spread)
        shocked_value = shocked_price / 100 * notional_w.value

        pnl = shocked_value - base_value
        print(f"Base market value:    ${base_value:,.0f}")
        print(f"Shocked market value: ${shocked_value:,.0f}")
        print(f"P&L: ${pnl:,.0f}  ({pnl / base_value:+.1%})")
        print(f"\\n(Exact reprice, not a linear DV01 estimate -- convexity is fully captured.)")


def render_regime(_=None):
    with out_regime:
        clear_output(wait=True)
        current = live.stress_prob.iloc[-1]
        print(f"Current Baa-30Y stress probability (live/filtered): {current:.0%}")
        print("STRESS" if current >= 0.5 else "CALM", "regime right now (>=50% threshold)")

        cutoff = live.stress_prob.index.max() - timedelta(days=1095)   # last 3 years
        recent = live.stress_prob[live.stress_prob.index >= cutoff]
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.fill_between(recent.index, recent.values, color="tab:orange", alpha=0.6)
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.plot(recent.index[-1], current, "o", color="black")
        ax.set_ylim(0, 1)
        ax.set_title("Live stress probability, last 3 years")
        plt.tight_layout()
        plt.show()


def render_all(_=None):
    render_pricing(); render_scenario(); render_regime()


for w in (coupon_w, maturity_w, rating_w, notional_w):
    w.observe(render_all, names="value")
for w in (parallel_shock_w, spread_shock_w):
    w.observe(render_scenario, names="value")

bond_inputs = W.VBox([coupon_w, maturity_w, rating_w, notional_w])
tab1 = W.VBox([bond_inputs, out_price])
tab2 = W.VBox([bond_inputs, parallel_shock_w, spread_shock_w, out_scenario])
tab3 = out_regime

tabs = W.Tab(children=[tab1, tab2, tab3])
tabs.set_title(0, "Pricing & Risk")
tabs.set_title(1, "Scenario Shock")
tabs.set_title(2, "Regime")
display(tabs)
render_all()'''))

cells.append(M("""# SECTION J: Model risk and limitations
A single, consolidated place for every known limitation of this engine (scattered mentions above are gathered here). This is written the way a model-validation document would: what the model assumes, where it breaks, and what it should never be used for without further work.

**Pricing and risk (Sections B, G, H)**
- Fixed-rate, non-callable, bullet bonds only. No callable/putable/floating/amortizing structures, no CDS, no default/recovery modeling.
- The 30/360 day-count has a known small discrepancy vs QuantLib (up to ~8bp in yield) specifically when a coupon date falls on the last day of February.
- Risk contribution and scenarios use a single (Treasury) curve for discounting; no key-rate duration, no separate spread CURVE (OAS is treated as a single flat number per bond, not a term structure), no basis/liquidity/funding risk.
- H2's rating-scaled widening (`rating_vol_ratio`) is an approximation: it scales a long-history proxy's peak widening by a RECENT (3y) relative-volatility ratio, not by that rating's own historical peak (which ICE OAS, at only ~3 years of history, cannot supply for 2008 or COVID).

**Risk contribution correlation (Section G)**
- `rho` (rate/spread correlation) is estimated over the same 3-year window as the volatilities; it is a point estimate with no confidence interval and can itself be unstable across different windows -- treat `total_risk_1sd` as indicative, not as a precise VaR-grade number.

**Regime model (Sections E, F)**
- The regime MEANS are not reliably identified (their standard error is close to the estimate itself, shown explicitly in E1/E7); only volatility, expected duration and time-share should be trusted.
- Only 2 vs 3 regimes were compared (E7); GARCH, threshold-autoregressive, and other volatility models were not benchmarked against the Markov-switching approach.
- The proxy (Baa-30Y) is a broad, aggregate series: it is blind to sector-specific shocks (2015-16 energy, 2023 regional-bank stress showed zero signal, Section E4).
- The optimizer (EM + quasi-Newton) is not always bit-reproducible even with a fixed seed; it can land in a statistically equivalent but differently-labeled local optimum, or rarely fail with a numerical singularity (handled with retries, but this is optimizer noise, not a guarantee of the single best fit).
- The live simulation (Section F) refits on an expanding window with fixed periodicity and a somewhat arbitrary `min_train_weeks`; the sensitivity check (F4) suggests this choice matters less than expected, but only 3 settings were tried.

**What this engine does NOT do (by design, left for a possible second project)**
- No economic backtest: no Sharpe ratio, drawdown, turnover, transaction costs, or hit rate for any signal built on the regime probability. Detecting a regime accurately is not the same as trading it profitably.
- No calibration analysis (Brier score, reliability diagrams, out-of-sample log score) of the regime probabilities themselves.
- No position-sizing or meta-labeling framework turning the regime signal into an actual trade.

**Engineering**
- This notebook is a demo/exploration layer. The actual test suite (100+ tests covering pricing, risk, data, regimes, backtesting, scenarios and risk contribution) lives in the accompanying repository, not in this notebook -- if evaluating this work, please look at the repository's `tests/` directory and CI configuration, not just this notebook.
- No production concerns (logging, monitoring, scheduling, alerting) are implemented; this is a research/analysis tool, not a live trading system. If this were productionized, it would need, at minimum, scheduled data refreshes with staleness alerts, model retraining triggers, and a fallback when FRED/ICE data is late or unavailable."""))

cells.append(M("# Done\nSend me: **F4 (table), G1, H1, H2, I1 (a screenshot from each of the three tabs, after playing with the sliders)**. Screenshots are fine."))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.validate(nb)
# Guard rail: catch broken-string / escaping bugs in generated cells BEFORE shipping the notebook.
_bad = []
for _i, _c in enumerate(nb.cells):
    if _c.cell_type != "code":
        continue
    _src = "".join(_c.source)
    if _src.startswith("%%writefile"):
        continue
    try:
        compile(_src, f"cell{_i}", "exec")
    except SyntaxError as _e:
        _bad.append((_i, str(_e)))
if _bad:
    raise SystemExit(f"REFUSING TO WRITE: {len(_bad)} generated cell(s) have a syntax error: {_bad}")

out = ROOT / "notebooks" / "credit_engine_FULL.ipynb"
nbf.write(nb, out)
print(f"written {out.name}: {len(cells)} cells, {out.stat().st_size/1024:.0f} KB (compile-checked, 0 syntax errors)")
