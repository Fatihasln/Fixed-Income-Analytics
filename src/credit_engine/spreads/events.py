"""Credit-market stress episodes for chart annotation.

Dates are APPROXIMATE and only used to label charts visually. They are not an
input to any model (regime models must not be told the answer).
"""
from __future__ import annotations

EVENTS: list[tuple[str, str, str]] = [
    ("Russia/LTCM 1998", "1998-08-01", "1998-11-30"),
    ("Dot-com credit cycle", "2001-03-01", "2002-10-31"),
    ("Global Financial Crisis", "2007-08-01", "2009-06-30"),
    ("US downgrade / euro crisis", "2011-08-01", "2011-12-31"),
    ("Energy & China 2015-16", "2015-06-01", "2016-02-29"),
    ("COVID-19", "2020-02-20", "2020-04-30"),
    ("Rate shock 2022", "2022-01-01", "2022-10-31"),
    ("SVB 2023", "2023-03-08", "2023-05-15"),
]


def plot_with_events(s_bp, title: str, events=EVENTS, figsize=(12, 5)):
    """Line chart of a spread in bp with shaded stress episodes (skips out-of-range ones)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    s = s_bp.dropna()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, lw=1.0, color="tab:blue")
    top = float(s.max())
    for name, a, b in events:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b < s.index[0] or a > s.index[-1]:
            continue
        ax.axvspan(a, b, color="tab:red", alpha=0.18, lw=0)
        ax.text(a + (b - a) / 2, top, name, rotation=90, va="top", ha="center", fontsize=8, color="darkred")
    ax.set_title(title)
    ax.set_ylabel("bp")
    ax.margins(x=0.01)
    fig.tight_layout()
    return fig
