from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the Yahoo historical prices, already expressed in $/gallon.
prices = pd.read_csv(
    DATA_DIR / "heating_oil_futures_2004_2009.csv",
    parse_dates=["date"]
)

prices = prices.sort_values("date").reset_index(drop=True)

valuation_date = pd.Timestamp("2009-04-24")

windows = [
    ("5 years", pd.Timestamp("2004-04-24"), "#24476B"),
    ("3 years", pd.Timestamp("2006-04-24"), "#C47A28"),
    ("1 year", pd.Timestamp("2008-04-25"), "#6B2447"),
]

prices = prices.loc[
    prices["date"].between(windows[0][1], valuation_date)
].copy()

if prices.empty:
    raise ValueError("No prices found in the requested period.")

if prices["date"].duplicated().any():
    raise ValueError("Duplicate price dates found.")

if (
    not np.isfinite(prices["price"]).all()
    or (prices["price"] <= 0).any()
):
    raise ValueError("Prices must be finite and positive.")


# Calculate each window's historical volatility.
# Returns are calculated AFTER selecting the window,
# matching our GARCH estimation-window comparison.
window_results = []

for label, start_date, color in windows:
    sample = prices.loc[
        prices["date"].between(start_date, valuation_date)
    ].copy()

    returns = np.log(
        sample["price"] / sample["price"].shift(1)
    ).dropna()

    if len(returns) < 2:
        raise ValueError(f"Insufficient returns for {label}.")

    annual_volatility = returns.std(ddof=1) * np.sqrt(252)

    window_results.append({
        "label": label,
        "first_date": sample["date"].iloc[0],
        "volatility": annual_volatility,
        "color": color,
    })


# Plot the price history.
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    prices["date"],
    prices["price"],
    color="#24476B",
    linewidth=1.3,
)

# Vertical markers show the first available price in each window.
# All three windows end on 24 April 2009.
for result in window_results:
    start = result["first_date"]
    color = result["color"]

    ax.axvline(
        start,
        color=color,
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
    )

    ax.text(
        start + pd.Timedelta(days=15),
        0.97,
        f"{result['label']} window starts\n"
        f"{start:%d %b %Y}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
        color=color,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 3,
        },
    )

# Put the volatility figures beneath the chart.
fig.text(
    0.11,
    0.125,
    "Historical annualized volatility",
    fontsize=10,
    fontweight="bold",
    color="#333333",
)

for x, result in zip([0.11, 0.40, 0.69], window_results):
    fig.text(
        x,
        0.080,
        f"{result['label']}: {result['volatility']:.2%}",
        fontsize=12,
        fontweight="bold",
        color=result["color"],
    )

ax.set_title(
    "Heating-Oil Futures Prices — Historical Estimation Windows",
    fontsize=14,
    pad=14,
)

ax.set_ylabel("Price ($/gal)")
ax.set_xlabel("Date")

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_formatter(StrMethodFormatter("${x:.2f}"))

ax.set_xlim(
    prices["date"].min() - pd.Timedelta(days=35),
    valuation_date + pd.Timedelta(days=45),
)

# Leave space above the price series for the window labels.
ax.set_ylim(
    0,
    prices["price"].max() * 1.22,
)

ax.grid(True, alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.11,
    0.025,
    "Source: Yahoo Finance (HO=F). All windows end 24 April 2009. "
    "Volatility: daily log-return standard deviation × √252.",
    fontsize=8,
    color="#555555",
)

fig.subplots_adjust(
    left=0.11,
    right=0.97,
    top=0.88,
    bottom=0.25,
)

for extension in ["png", "pdf"]:
    fig.savefig(
        OUTPUT_DIR / f"historical_price_windows.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.show()
plt.close(fig)

print(f"Graphs saved to: {OUTPUT_DIR}")