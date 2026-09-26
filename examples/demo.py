"""Day-1 demo: risk report for three *illustrative* bonds (not real issues).

Run:  python examples/demo.py
"""
import sys
from datetime import date
from pathlib import Path

# Works even if the package is not pip-installed (e.g. plain notebook / fresh clone).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_engine.pricing.bond import Bond
from credit_engine.pricing.curves import flat_curve
from credit_engine.pricing.bond import dirty_price_from_curve
from credit_engine.risk.cs01 import cs01, dv01_curve, z_spread
from credit_engine.risk.duration import rate_risk_report

SETTLE = date(2026, 9, 21)
TSY = 0.042  # flat illustrative Treasury zero rate

# (label, coupon, maturity, spread over Treasuries) - illustrative numbers only
BONDS = [
    ("AAA-like  5y", Bond(0.045, date(2031, 9, 15)), 0.0060),
    ("BBB-like 10y", Bond(0.058, date(2036, 9, 15)), 0.0150),
    ("HY-like   7y", Bond(0.085, date(2033, 9, 15)), 0.0450),
]

print(f"Settlement {SETTLE}, flat Treasury zero {TSY:.2%}, notional 10mm\n")
print(f"{'bond':<14}{'YTM':>7}{'clean':>9}{'MacD':>7}{'ModD':>7}{'Conv':>8}{'DV01($)':>10}{'CS01($)':>10}")
for label, bond, spread in BONDS:
    curve = flat_curve(TSY)
    dirty = dirty_price_from_curve(bond, SETTLE, curve, spread)
    ytm = TSY + spread
    r = rate_risk_report(bond, SETTLE, ytm, notional=10_000_000)
    assert abs(r.dirty_price - dirty) < 1e-9
    assert abs(z_spread(bond, SETTLE, dirty, curve) - spread) < 1e-10
    c = cs01(bond, SETTLE, curve, spread, notional=10_000_000)
    print(f"{label:<14}{ytm:>7.2%}{r.clean_price:>9.3f}{r.macaulay:>7.3f}{r.modified:>7.3f}"
          f"{r.convexity:>8.2f}{r.dv01:>10,.0f}{c:>10,.0f}")
