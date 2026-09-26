"""Stress scenarios: reprice a portfolio of bonds under a curve/spread shock and report P&L.

Three built-in scenario families:
  * parallel(bp)                  -- flat shift of the whole curve
  * steepen(short_bp, long_bp)    -- short end and long end move by different amounts (linear in between)
  * spread_widening(bp)           -- credit spread widens (or tightens, negative bp), curve unchanged (uniform
                                     across every bond -- see the WARNING below)
  * scaled_historical_scenarios   -- spread widening sized PER RATING BUCKET, not uniformly (see below)

Each scenario reprices every bond exactly (via dirty_price_from_curve), not via a duration approximation,
so results are exact even for large shocks where convexity would make a linear DV01 estimate misleading.

WARNING about uniform bp widening: applying the SAME bp spread widening to every bond regardless of its rating
is unrealistic. In a real crisis, high-yield spreads widen far more than AAA spreads (e.g. in 2008, HY OAS
widened by several thousand bp while AAA widened by a few hundred). ``spread_widening`` and
``historical_style_scenarios`` do this uniformly and are kept for simple, single-bucket analyses; for a
multi-rating portfolio, use ``scaled_historical_scenarios`` instead, which scales the widening per bond using
``portfolio.risk_contribution.rating_vol_ratio`` (each bond's own rating volatility relative to the reference
series the widening was measured on).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..pricing.bond import Bond, ZeroCurve, dirty_price_from_curve
from ..pricing.curves import linear_curve

SpreadShift = float | dict[str, float]  # uniform (bp, applied to every bond) or per-bond ({label: bp})


@dataclass(frozen=True)
class PortfolioBond:
    label: str
    bond: Bond
    spread: float          # credit spread over the base curve, decimal
    notional: float        # par notional


@dataclass(frozen=True)
class Scenario:
    name: str
    curve_shift: "callable"      # years -> additional decimal shift to the base curve
    spread_shift_bp: SpreadShift  # additional spread shift, in bp: a single float (uniform) or {label: bp}


def _shift_for(spread_shift_bp: SpreadShift, label: str) -> float:
    """Resolve the bp shift for one bond, whether the scenario carries a uniform value or a per-bond dict."""
    if isinstance(spread_shift_bp, dict):
        if label not in spread_shift_bp:
            raise KeyError(f"scenario has no spread shift for bond {label!r}; has: {sorted(spread_shift_bp)}")
        return spread_shift_bp[label]
    return spread_shift_bp


def parallel(bp: float) -> Scenario:
    return Scenario(f"Parallel {bp:+.0f}bp", lambda years: np.full_like(np.asarray(years, dtype=float), bp / 1e4), 0.0)


def steepen(short_bp: float, long_bp: float, pivot_years: tuple[float, float] = (2.0, 30.0)) -> Scenario:
    """Linear interpolation of the shift between ``pivot_years``; flat beyond the pivots."""
    shift_curve = linear_curve(list(pivot_years), [short_bp / 1e4, long_bp / 1e4])
    name = f"Steepener (2Y {short_bp:+.0f}bp / 30Y {long_bp:+.0f}bp)"
    return Scenario(name, shift_curve, 0.0)


def spread_widening(bp: float, label: str | None = None) -> Scenario:
    """Uniform widening: every bond's spread moves by the SAME bp. See the module WARNING above."""
    name = label or f"Spread {bp:+.0f}bp"
    return Scenario(name, lambda years: np.zeros_like(np.asarray(years, dtype=float)), bp)


def apply_scenario(base_curve: ZeroCurve, scenario: Scenario) -> ZeroCurve:
    return lambda years: np.asarray(base_curve(years), dtype=float) + np.asarray(scenario.curve_shift(years), dtype=float)


def price_portfolio(portfolio: list[PortfolioBond], settlement: date, curve: ZeroCurve,
                    spread_bump: SpreadShift = 0.0) -> float:
    """Total dirty market value of the portfolio (per-bond price/100 * notional), summed.

    ``spread_bump`` (bp) is either one number applied to every bond, or a {label: bp} dict for per-bond bumps.
    """
    total = 0.0
    for pb in portfolio:
        bump = _shift_for(spread_bump, pb.label)
        price = dirty_price_from_curve(pb.bond, settlement, curve, pb.spread + bump / 1e4)
        total += price / 100.0 * pb.notional
    return total


def run_scenarios(portfolio: list[PortfolioBond], settlement: date, base_curve: ZeroCurve,
                  scenarios: list[Scenario]) -> pd.DataFrame:
    """P&L (exact reprice, not duration-approximated) for the whole portfolio and per-bond, for each scenario."""
    base_value = price_portfolio(portfolio, settlement, base_curve)
    per_bond_base = {pb.label: dirty_price_from_curve(pb.bond, settlement, base_curve, pb.spread) / 100.0 * pb.notional
                     for pb in portfolio}

    rows = []
    for sc in scenarios:
        shocked_curve = apply_scenario(base_curve, sc)
        shocked_value = price_portfolio(portfolio, settlement, shocked_curve, sc.spread_shift_bp)
        row = {"scenario": sc.name, "portfolio_pnl_$": shocked_value - base_value,
              "portfolio_pnl_pct": (shocked_value - base_value) / base_value * 100}
        for pb in portfolio:
            bump = _shift_for(sc.spread_shift_bp, pb.label)
            shocked_bond_value = dirty_price_from_curve(pb.bond, settlement, shocked_curve,
                                                        pb.spread + bump / 1e4) / 100.0 * pb.notional
            row[f"{pb.label}_pnl_$"] = shocked_bond_value - per_bond_base[pb.label]
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def historical_style_scenarios(peak_spread_bp: dict[str, float], current_spread_bp: float) -> list[Scenario]:
    """Build UNIFORM spread-widening scenarios sized to historical episodes, e.g. {"2008 GFC": 620,
    "COVID-19": 430} (peak Baa-30Y spread in bp, from regimes.regime_periods) vs. today's level.

    See the module WARNING: this applies the SAME bp widening to every bond regardless of rating. Prefer
    ``scaled_historical_scenarios`` for a multi-rating portfolio.
    """
    return [spread_widening(peak - current_spread_bp, label=f"{name}-style widening (to {peak:.0f}bp)")
           for name, peak in peak_spread_bp.items()]


def scaled_historical_scenarios(peak_spread_bp: dict[str, float], current_spread_bp: float,
                                rating_ratios: dict[str, float]) -> list[Scenario]:
    """Historical-style widening, scaled PER BOND by ``rating_ratios`` (label -> volatility ratio vs. the
    reference series the peaks were measured on, e.g. from ``risk_contribution.rating_vol_ratio``).

    The base widening (peak - current, on the long-history reference proxy) is multiplied by each bond's own
    ratio, so a high-yield bond (ratio > 1, more volatile than the broad reference) widens by MORE than the
    reference figure and an AAA bond (ratio < 1) widens by LESS -- unlike ``historical_style_scenarios``, which
    would apply the same raw widening to both.
    """
    base_widenings = {name: peak - current_spread_bp for name, peak in peak_spread_bp.items()}
    scenarios = []
    for name, base_bp in base_widenings.items():
        per_bond_bp = {label: base_bp * ratio for label, ratio in rating_ratios.items()}
        scenarios.append(Scenario(f"{name}-style widening (scaled by rating)", 
                                  lambda years: np.zeros_like(np.asarray(years, dtype=float)), per_bond_bp))
    return scenarios
