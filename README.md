# credit-analytics-engine

Corporate bond pricing, interest-rate risk (Macaulay / Modified duration, convexity, DV01),
credit-spread risk (CS01, Z-spread) and, in later versions, spread regime analysis.

**Status: v0.8 - correlation-adjusted total risk, real (non-flat) Treasury curve, rating-scaled historical scenarios, 2-vs-3-regime AIC/BIC comparison, regime parameter standard errors, CI (GitHub Actions).**

## What is implemented
| Module | Contents |
|---|---|
| `pricing/daycount.py` | 30/360 US, ACT/ACT ICMA |
| `pricing/bond.py` | schedule, cash flows, accrued, clean/dirty price, curve+spread pricing |
| `pricing/yield_solver.py` | YTM via Brent (robust) and Newton (analytic slope) |
| `pricing/curves.py` | flat / piecewise-linear zero curve, parallel shift |
| `risk/duration.py` | Macaulay, Modified, convexity, DV01, effective duration/convexity (bump-and-reprice) |
| `risk/cs01.py` | CS01, curve DV01, spread duration, Z-spread |
| `data/fred.py` | FRED download (retries, disk cache, sanity checks); API key optional |
| `spreads/spread_calc.py` | spread construction, data-gap detection, proxy-vs-reference validation |
| `spreads/events.py` | stress-episode annotation for charts (visual only, never a model input) |
| `spreads/regimes.py` | 2-regime Markov Switching (calm/stress) on weekly log-changes; filtered vs smoothed probabilities |
| `spreads/backtest.py` | expanding-window refit: simulates what a live system would have known at each date |
| `portfolio/risk_contribution.py` | sensitivity (DV01/CS01) x realized weekly volatility -- makes the Section 2.3 finding quantitative |
| `portfolio/scenarios.py` | exact reprice (not duration-approximated) under parallel/steepener/spread-widening shocks, incl. historical-style scenarios |

## Data sources and licensing
* FRED (St. Louis Fed). Since April 2026 FRED serves only ~3 years of the ICE BofA OAS series, so
  long-history regime analysis uses Moody's Baa/Aaa minus 10Y Treasury (daily, 1986+) as a proxy.
  ICE HY OAS (last ~3 years) is used only for current spread levels and volatility.
* Moody's and ICE data must not be redistributed: raw data is **never committed** (`data/raw/` is git-ignored);
  each user downloads it with `notebooks/credit_engine_FULL.ipynb` or `credit_engine.data.fred`.

### API keys
Never write a FRED key into a notebook or commit it. In Colab use *Secrets* (name `FRED_API_KEY`); locally use the
`FRED_API_KEY` environment variable. The key is optional: the loader also works without it.

## Conventions
* Prices per 100 par; yields periodically compounded at the coupon frequency (street convention).
* Duration, convexity, DV01 are measured on the **dirty** price.
* DV01 / CS01 are positive numbers = loss for a +1bp move.
* Coupon dates are generated backwards from maturity, no business-day adjustment.
* Days to next coupon = period length - accrued days (SIA standard).

## Validation
`pytest` runs 50+ tests, including:
* a worked-by-hand example (2y, 5% annual, par) with every number in the test docstring,
* analytic duration/convexity vs bump-and-reprice,
* second-order Taylor error scaling (verifies convexity is correct, not just plausible),
* 300 random bonds x 2 day counts vs **QuantLib** (price, accrued, yield, Macaulay, Modified, convexity, DV01).

## Known limitations
* **30/360 with a coupon date on the last day of February:** QuantLib's 30/360 US treatment differs
  slightly from ours (median ~1bp, max ~8bp in yield on 36 random cases). Tracked by
  `test_feb_eom_30_360_known_deviation`. ACT/ACT is unaffected.
* Bullet, fixed-rate, non-callable bonds only. No optionality (so effective duration == modified duration).
* No business-day adjustment or ex-coupon periods.

## A note on CS01 vs DV01
For a plain bullet bond and a *parallel* shift, zero rate and spread enter the discount factor
additively, so **CS01 equals curve DV01 exactly** (tested). The two risks differ in *which factor
moves and by how much*: spread volatility is much larger than rate volatility for high yield and in
stress. The meaningful comparison is therefore `sensitivity x typical factor move`
(risk contribution), which is what the Day-2 analysis builds on real FRED data.

## Quick start
```bash
pip install -e ".[dev]"     # installs the package + pytest + QuantLib
pytest
python examples/demo.py
```

### Jupyter / Colab
The code lives in `src/`, so the package must be installed or `src` added to the path:
```python
!unzip -q credit-analytics-engine-day1.zip && cd credit-analytics-engine && pip install -e ".[dev]"
# if the import still fails, restart the runtime, or in the first cell:
import sys; sys.path.insert(0, "credit-analytics-engine/src")
```

## Regime model: what is (and is not) identified
Fit on weekly log-changes of the spread, 2 regimes (calm/stress), both mean and variance switch.
* **Only the FILTERED probability is safe to use as a signal** (uses data up to t only). The SMOOTHED probability
  uses the whole sample and is for diagnostics/plots only. Tested: extending the sample changes smoothed values
  for the past by up to 0.8 in probability; filtered values are provably unaffected (diff < 1e-9).
* **Regime means are not reliably identified**: their standard errors are close to the size of the means
  themselves (few effective stress weeks). Interpret volatility, expected duration and time-share instead.
* **Optimiser is not always bit-reproducible** even with a fixed seed (multi-threaded BLAS can land in a
  different, statistically-equivalent, label-swapped local optimum, and can occasionally fail with a numerical
  singularity). The code retries with a shrinking search on failure; this is documented, not silently patched.
* `backtest.expanding_refit` refits periodically (default: every 13 weeks) on an expanding window and stitches
  filtered probabilities, to see what a live system would actually have known -- typically less confident and
  slower to react than the in-sample (whole-history) fit.

## Risk contribution and stress scenarios
`risk_contribution.py` turns the DV01=CS01 identity (Section 2.3) into a real comparison: it multiplies each
sensitivity by the REALIZED weekly volatility of its factor (Treasury yield / credit spread, from FRED data),
so "which risk dominates" is answered with actual numbers, not assumed. `scenarios.py` reprices a portfolio
exactly (not via a linear DV01 estimate) under parallel shifts, curve steepeners and spread-widening shocks,
including scenarios sized to historical episodes (e.g. "widen to the 2008 peak"). An `ipywidgets`-based
interactive panel (Colab notebook, Section I) ties pricing, risk contribution and the live regime probability
together for an arbitrary bond typed in on the spot.

## A real bug found from actual output, not just review
Running Section G on real data initially gave the SAME `spread_share_pct` for every bond regardless of rating.
Root cause: every bond was passed the same broad-market spread series. Since DV01==CS01 for a parallel shift
(Section 2.3), the split percentage collapses to `spread_vol / (rate_vol + spread_vol)` -- identical for every
bond if the same two series are reused, regardless of the bond itself. Fixed by mapping each rating to its own
ICE OAS bucket (`RATING_TO_ICE_BUCKET` / `spread_series_for_rating`) and by windowing volatility to the last
3 years (`window_years`) so the comparison reflects the current regime, not a 40+ year blended average.

## Correlation and total portfolio risk
`rate_risk_contribution` and `spread_risk_contribution` are MARGINAL pieces: they do not sum to the true
portfolio P&L variance unless rate and spread moves are uncorrelated, which they usually are not (e.g. a
flight-to-quality move: rates fall while spreads widen, a negative correlation). The correct total is

    Var(P&L) = DV01^2 * sigma_r^2 + CS01^2 * sigma_s^2 + 2 * DV01 * CS01 * rho * sigma_r * sigma_s

`risk_contribution.factor_correlation` estimates rho from realized weekly changes, and `total_risk_1sd` (vs.
the naive, uncorrelated `naive_sum_1sd`) reports the corrected total. `dv01` itself now comes from
`risk.cs01.dv01_curve` (a real curve bump), not a flat-YTM approximation, so it stays correct on a sloped
Treasury curve.

## Rating-scaled historical scenarios
Applying the SAME bp spread widening to every bond in a historical-crisis scenario is unrealistic: HY spreads
widen far more than AAA spreads in a real crisis. `scenarios.scaled_historical_scenarios` scales each bond's
widening by `risk_contribution.rating_vol_ratio` (its own rating bucket's realized volatility relative to the
long-history proxy the historical peak was measured on).

## Regime model selection and standard errors
`regimes.compare_regime_counts` fits the model with different numbers of hidden regimes (e.g. 2 vs 3) and
compares by AIC/BIC, rather than assuming 2 regimes is correct. `fit_regime_model`'s result now also carries
`param_se` (parameter standard errors) and the regime summary includes `mean_se_pct` / `weekly_sigma_se_pct`,
making explicit (not just asserted) that the regime means are usually not statistically significant while the
volatilities are precisely estimated.

## Model risk and limitations
A single consolidated list of every known limitation (pricing/instrument scope, curve simplifications, regime
model caveats, and what this project deliberately does NOT do -- economic backtesting, calibration, signal
generation) lives in Section J of the Colab notebook. Read that before drawing conclusions from any single
number in this repository.

## Continuous integration
`.github/workflows/tests.yml` runs the full pytest suite (Python 3.10/3.11/3.12) and regenerates the Colab
notebook (which fails the build if any generated cell has a syntax error) on every push and pull request.
