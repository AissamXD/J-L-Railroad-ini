from pricing import black76_call, black76_put
from asian_bootstrap import (
    prepare_bootstrap_returns,
    build_daily_forward_curve,
    simulate_bootstrap_paths,
    price_asian_cap_floor,
    scale_returns_to_volatility
)
from asian_garch import (
    fit_garch,
    simulate_garch_average_prices
)
from market_check import (
    load_market_options,
    calculate_market_implied_volatilities
)
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_prices():
    """Load and validate the historical heating-oil prices."""

    file_path = DATA_DIR / "historical_prices.csv"

    prices = pd.read_csv(file_path, parse_dates=["date"])
    prices = prices.sort_values("date").reset_index(drop=True)

    # Convert cents per gallon into dollars per gallon
    prices["price"] = prices["close_cents"] / 100

    if prices["date"].duplicated().any():
        raise ValueError("Duplicate dates found in historical_prices.csv")

    if prices["price"].isna().any():
        raise ValueError("Missing prices found in historical_prices.csv")

    if (prices["price"] <= 0).any():
        raise ValueError("Prices must be positive.")

    return prices


def calculate_volatility(prices):
    """Calculate annualized historical volatility from daily log returns."""

    prices = prices.copy()

    prices["log_return"] = np.log(
        prices["price"] / prices["price"].shift(1)
    )

    returns = prices["log_return"].dropna()

    daily_vol = returns.std(ddof=1)
    annual_vol = daily_vol * np.sqrt(252)

    return daily_vol, annual_vol, prices

def load_forward_curve():
    """Load and validate the Exhibit 3 futures curve."""

    file_path = DATA_DIR / "forward_curve.csv"

    curve = pd.read_csv(
        file_path,
        parse_dates=["delivery_month"]
    )

    curve = curve.sort_values(
        "delivery_month"
    ).reset_index(drop=True)

    if curve["delivery_month"].isna().any():
        raise ValueError("Missing delivery months in forward_curve.csv")

    if curve["delivery_month"].duplicated().any():
        raise ValueError("Duplicate delivery months in forward_curve.csv")

    if not np.isfinite(curve["futures_price"]).all():
        raise ValueError("Missing or infinite futures prices")

    if (curve["futures_price"] <= 0).any():
        raise ValueError("Futures prices must be positive")

    expected_months = pd.date_range(
        start="2009-05-01",
        end="2010-05-01",
        freq="MS"
    )

    if not pd.DatetimeIndex(curve["delivery_month"]).equals(
        expected_months
    ):
        raise ValueError(
            "Expected 13 delivery months: May 2009 through May 2010"
        )

    return curve

def load_extended_prices():
    """Load the longer Yahoo heating-oil futures history."""

    file_path = DATA_DIR / "heating_oil_futures_2004_2009.csv"

    prices = pd.read_csv(
        file_path,
        parse_dates=["date"]
    )

    prices["price"] = pd.to_numeric(
        prices["price"],
        errors="raise"
    )

    if prices["date"].isna().any():
        raise ValueError("Missing dates in the extended history.")

    if prices["date"].duplicated().any():
        raise ValueError("Duplicate dates in the extended history.")

    # Exclude any observations outside our historical window.
    prices = prices.loc[
        prices["date"].between(
            "2004-04-24",
            "2009-04-24"
        )
    ].copy()

    prices = prices.sort_values("date").reset_index(drop=True)

    if prices.empty:
        raise ValueError("No prices found in the selected window.")

    if not np.isfinite(prices["price"]).all():
        raise ValueError("Missing or infinite prices.")

    if (prices["price"] <= 0).any():
        raise ValueError("Prices must be positive.")

    # Already in dollars per gallon: no division by 100.
    return prices

def main():
    print("J&L Railroad — Pricing and Market Check")
    print("-" * 45)

    prices = load_prices()

    daily_vol, annual_vol, prices = calculate_volatility(prices)

    #Asian bootstrap" prepare the empirical daily shocks.
    #The first lo return is missing because there is no previous price
    historical_returns = prices["log_return"].dropna().to_numpy()

    bootstrap_returns, drift_adjustment = prepare_bootstrap_returns(historical_returns)

    historical_multiplier = np.exp(historical_returns).mean()
    adjusted_multiplier = np.exp(bootstrap_returns).mean()

    print("\nASIAN BOOTSTRAP - HISTORICAL SHOCKS")
    print("-" * 50)
    print(f"Number of historical returns: {len(historical_returns)}")
    print("Historical average gross multiplier: " f"{historical_multiplier:.10f}")
    print("Adjusted average gross multiplier: " f"{adjusted_multiplier:.10f}")
    print("Daily log-return adjustment: " f"{drift_adjustment:.8f}")
    print("Original daily standard deviation: " f"{historical_returns.std(ddof=1):.6%}")
    print("Adjusted daily standard deviation: " f"{bootstrap_returns.std(ddof=1):.6%}")

    #Confirms that the bootstrap adjustment is correct
    
    if not np.isclose(
        adjusted_multiplier,
        1.0,
        rtol=0.0,
        atol=1e-12
    ):
        raise ValueError(
            "The adjusted average gross multiplier must equal 1."
        )
    forward_curve = load_forward_curve()
    print("\nASIAN BOOTSTRAP — FORWARD CURVE")
    print("-" * 50)
    print(forward_curve.to_string(index=False))

    # Diagnostic: equal-weight average of May 2009–April 2010.
    first_year_average = (
        forward_curve["futures_price"].iloc[:12].mean()
    )

    print(
        "\nEqual-weight average of first 12 futures prices: "
        f"${first_year_average:.6f}/gallon"
    )

    # Build the daily baseline from the monthly futures curve.
    daily_forwards = build_daily_forward_curve(
        forward_curve,
        n_steps=252
    )

    print("\nASIAN BOOTSTRAP — DAILY BASELINE")
    print("-" * 50)

    print(f"Number of observations: {len(daily_forwards)}")

    print(
        "First baseline price: "
        f"${daily_forwards[0]:.6f}/gal"
    )

    print(
        "Last baseline price: "
        f"${daily_forwards[-1]:.6f}/gal"
    )

    print(
        "Average daily baseline: "
        f"${daily_forwards.mean():.6f}/gal"
    )

    # Generate 10,000 price paths, each with 252 observations.
    simulated_prices = simulate_bootstrap_paths(
        daily_forwards=daily_forwards,
        bootstrap_returns=bootstrap_returns,
        n_paths=50000,
        seed=42
    )

    # Calculate the annual arithmetic-average price for each path.
    average_prices = simulated_prices.mean(axis=1)

    # Check whether the simulation reproduces the baseline average.
    baseline_average = daily_forwards.mean()
    simulated_average = average_prices.mean()

    # Monte Carlo standard error of the estimated mean.
    mean_standard_error = (
        average_prices.std(ddof=1)
        / np.sqrt(len(average_prices))
    )

    print("\nASIAN BOOTSTRAP — SIMULATED PATHS")
    print("-" * 50)

    print(f"Number of paths: {simulated_prices.shape[0]}")
    print(f"Observations per path: {simulated_prices.shape[1]}")

    print(
        "Baseline annual average: "
        f"${baseline_average:.6f}/gallon"
    )

    print(
        "Mean of simulated annual averages: "
        f"${simulated_average:.6f}/gallon"
    )

    print(
        "Monte Carlo standard error of mean: "
        f"${mean_standard_error:.6f}/gallon"
    )

    print(
        "Difference from baseline: "
        f"${simulated_average - baseline_average:+.6f}/gallon"
    )

    print(
        "5th percentile of annual averages: "
        f"${np.quantile(average_prices, 0.05):.4f}/gallon"
    )

    print(
        "95th percentile of annual averages: "
        f"${np.quantile(average_prices, 0.95):.4f}/gallon"
    )

    # Asian contract inputs.
    asian_cap_strike = 1.68
    asian_floor_strike = 1.40
    asian_rate = 0.0049

    # Assumption: payment on April 30, 2010.
    # Discount from the April 24, 2009 market-data date.
    valuation_date = pd.Timestamp("2009-04-24")
    payment_date = pd.Timestamp("2010-04-30")

    asian_discount_years = (
        (payment_date - valuation_date).days / 365
    )

    asian_results = price_asian_cap_floor(
        average_prices=average_prices,
        cap_strike=asian_cap_strike,
        floor_strike=asian_floor_strike,
        risk_free_rate=asian_rate,
        discount_years=asian_discount_years
    )

    # Exhibit 2: interpolate the cap quote at strike $1.68.
    asian_cap_quote = (
        0.088
        + (1.68 - 1.65) / (1.70 - 1.65)
        * (0.073 - 0.088)
    )

    asian_floor_quote = 0.079

    quotes = {
        "Cap": asian_cap_quote,
        "Floor": asian_floor_quote,
        "Net collar": asian_cap_quote - asian_floor_quote,
    }

    rows = []

    for option, result in asian_results.items():
        rows.append({
            "Option": option,
            "KCNB quote": quotes[option],
            "Bootstrap value": result["value"],
            "MC standard error": result["standard_error"],
            "95% MC lower": result["ci_lower"],
            "95% MC upper": result["ci_upper"],
            "Quote minus model": (
                quotes[option] - result["value"]
            ),
        })

    asian_comparison = pd.DataFrame(rows)

    print("\nASIAN BOOTSTRAP — PRICE COMPARISON ($/gallon)")
    print("-" * 110)
    print(asian_comparison.round(5).to_string(index=False))

    asian_comparison.to_csv(
        OUTPUT_DIR / "asian_bootstrap_comparison.csv",
        index=False
    )

    # Chart: KCNB quotes versus Asian bootstrap valuations.
    # Multiply dollar values by 100 to display cents per gallon.

    model_color = "#24476B"
    quote_color = "#C47A28"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5.5),
        gridspec_kw={"width_ratios": [1.6, 1]}
    )

    panels = [
        (axes[0], ["Cap", "Floor"], "Individual option legs"),
        (axes[1], ["Net collar"], "Net collar: cap minus floor"),
    ]

    for ax, names, panel_title in panels:
        x = np.arange(len(names))

        model_values = np.array([
            asian_results[name]["value"] * 100
            for name in names
        ])

        quote_values = np.array([
            quotes[name] * 100
            for name in names
        ])

        error_margins = np.array([
            1.96 * asian_results[name]["standard_error"] * 100
            for name in names
        ])

        # Bootstrap estimate with a 95% Monte Carlo interval.
        ax.errorbar(
            x - 0.12,
            model_values,
            yerr=error_margins,
            fmt="o",
            markersize=8,
            color=model_color,
            ecolor=model_color,
            elinewidth=2,
            capsize=6,
            label="Asian bootstrap",
            zorder=3
        )

        # KCNB quoted premium.
        ax.scatter(
            x + 0.12,
            quote_values,
            marker="D",
            s=65,
            color=quote_color,
            label="KCNB quote",
            zorder=4
        )

        # Label model estimates above their intervals.
        for position, value, margin in zip(
            x, model_values, error_margins
        ):
            ax.annotate(
                f"{value:.2f}",
                (position - 0.12, value + margin),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=model_color,
                fontsize=10,
                fontweight="bold"
            )

        # Label quotes below their markers.
        for position, value in zip(x, quote_values):
            ax.annotate(
                f"{value:.2f}",
                (position + 0.12, value),
                xytext=(0, -17),
                textcoords="offset points",
                ha="center",
                color=quote_color,
                fontsize=10,
                fontweight="bold"
            )

        ax.axhline(
            0,
            color="#666666",
            linewidth=0.9,
            linestyle="--"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlim(-0.6, len(names) - 0.4)

        ax.set_title(panel_title, fontsize=12, pad=14)
        ax.set_ylabel("Value (cents per gallon)")

        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Leave room for labels on either side of the plotted values.
        lower = min(
            0.0,
            np.min(model_values - error_margins),
            np.min(quote_values)
        )

        upper = max(
            0.0,
            np.max(model_values + error_margins),
            np.max(quote_values)
        )

        span = max(upper - lower, 0.1)
        ax.set_ylim(
            lower - 0.22 * span,
            upper + 0.25 * span
        )

    fig.suptitle(
        "KCNB quotes versus Asian bootstrap valuations",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False
    )

    fig.text(
        0.5,
        0.075,
        f"{len(average_prices):,} simulated paths | "
        "Error bars: 95% Monte Carlo intervals | "
        "Panels use different vertical scales",
        ha="center",
        fontsize=9,
        color="#444444"
    )

    fig.text(
        0.5,
        0.03,
        "Intervals show simulation uncertainty only. "
        "The $1.68 cap quote is interpolated from Exhibit 2.",
        ha="center",
        fontsize=9,
        color="#444444"
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.84])

    fig.savefig(
        OUTPUT_DIR / "asian_bootstrap_vs_kcnb.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT_DIR / "asian_bootstrap_vs_kcnb.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    # Asian bootstrap: sensitivity to annual volatility.
    asian_volatility_scenarios = [
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
    ]

    # Use the same number of paths as the baseline simulation.
    sensitivity_n_paths = len(average_prices)

    asian_sensitivity_rows = []

    for target_sigma in asian_volatility_scenarios:
        scenario_returns = scale_returns_to_volatility(
            historical_returns=historical_returns,
            target_annual_volatility=target_sigma
        )

        # Reusing the seed selects the same historical observations
        # in every scenario. Only their shock sizes change.
        scenario_paths = simulate_bootstrap_paths(
            daily_forwards=daily_forwards,
            bootstrap_returns=scenario_returns,
            n_paths=sensitivity_n_paths,
            seed=42
        )

        scenario_averages = scenario_paths.mean(axis=1)

        # Keep memory use lower before the next simulation.
        del scenario_paths

        scenario_results = price_asian_cap_floor(
            average_prices=scenario_averages,
            cap_strike=asian_cap_strike,
            floor_strike=asian_floor_strike,
            risk_free_rate=asian_rate,
            discount_years=asian_discount_years
        )

        cap_value = scenario_results["Cap"]["value"]
        floor_value = scenario_results["Floor"]["value"]

        asian_sensitivity_rows.append({
            "Volatility": target_sigma,
            "Cap Value ($/gal)": cap_value,
            "Floor Value ($/gal)": floor_value,
            "Net Collar Value ($/gal)": (
                scenario_results["Net collar"]["value"]
            ),
            "Cap MC SE ($/gal)": (
                scenario_results["Cap"]["standard_error"]
            ),
            "Floor MC SE ($/gal)": (
                scenario_results["Floor"]["standard_error"]
            ),
            "Cap quote minus model ($/gal)": (
                asian_cap_quote - cap_value
            ),
            "Floor quote minus model ($/gal)": (
                asian_floor_quote - floor_value
            ),
        })

    asian_sensitivity = pd.DataFrame(asian_sensitivity_rows)

    print("\nASIAN BOOTSTRAP — VOLATILITY SENSITIVITY")
    print("-" * 110)
    print(asian_sensitivity.round(5).to_string(index=False))

    asian_sensitivity.to_csv(
        OUTPUT_DIR / "asian_volatility_sensitivity.csv",
        index=False
    )

    
    plt.figure(figsize=(9, 5))

    plt.plot(
        asian_sensitivity["Volatility"] * 100,
        asian_sensitivity["Cap Value ($/gal)"],
        label="Cap Value",
        marker="o",
        color="#24476B"
    )

    plt.plot(
        asian_sensitivity["Volatility"] * 100,
        asian_sensitivity["Floor Value ($/gal)"],
        label="Floor Value",
        marker="o",
        color="#C47A28"
    )

    plt.axhline(
        asian_cap_quote,
        linestyle="--",
        color="#28644A",
        label="KCNB Cap Quote",
        zorder=3
    )

    plt.axhline(
        asian_floor_quote,
        linestyle=":",
        color="#6B2447",
        label="KCNB Floor Quote",
        zorder=4
    )

    plt.title("Volatility Sensitivity of Asian Option Valuation")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Option Value ($/gal)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "asian_volatility_sensitivity.png",
        dpi=300
    )

    plt.savefig(
        OUTPUT_DIR / "asian_volatility_sensitivity.pdf"
    )

    plt.close()

        # Fit GARCH to the original historical log returns.
    garch_result = fit_garch(historical_returns)

    print("\nGARCH(1,1) — ESTIMATION")
    print("-" * 70)
    print(garch_result.summary())

    omega = float(garch_result.params["omega"])
    alpha = float(garch_result.params["alpha[1]"])
    beta = float(garch_result.params["beta[1]"])

    persistence = alpha + beta

    # Forecast variance for each of the next 252 model days.
    garch_forecast = garch_result.forecast(
        horizon=252,
        method="analytic",
        reindex=False
    )

    # Forecast variances are in percentage-return units squared.
    forecast_variances_pct = (
        garch_forecast.residual_variance
        .iloc[-1]
        .to_numpy()
    )

    # Convert to decimal variances.
    forecast_variances = forecast_variances_pct / 10000

    # Annualize daily standard deviations.
    forecast_annual_vols = np.sqrt(
        forecast_variances * 252
    )

    # Square root of average forecast daily variance, annualized.
    average_forecast_vol = np.sqrt(
        forecast_variances.mean() * 252
    )

    print("\nGARCH — VOLATILITY DIAGNOSTICS")
    print("-" * 70)
    print(f"Alpha: {alpha:.6f}")
    print(f"Beta: {beta:.6f}")
    print(f"Persistence (alpha + beta): {persistence:.6f}")

    print(
        "Historical annualized volatility: "
        f"{annual_vol:.2%}"
    )

    print(
        "Next-day forecast volatility, annualized: "
        f"{forecast_annual_vols[0]:.2%}"
    )

    print(
        "Day-252 forecast volatility, annualized: "
        f"{forecast_annual_vols[-1]:.2%}"
    )

    print(
        "Annualized volatility from average forecast variance: "
        f"{average_forecast_vol:.2%}"
    )

    if persistence < 1:
        long_run_daily_variance = (
            omega / (1 - persistence)
        ) / 10000

        long_run_annual_vol = np.sqrt(
            long_run_daily_variance * 252
        )

        print(
            "Fitted long-run annualized volatility: "
            f"{long_run_annual_vol:.2%}"
        )
    else:
        print(
            "No finite long-run variance: "
            "alpha + beta is at least 1."
        )
    extended_prices = load_extended_prices()
    
    # Compare GARCH estimates across historical windows.
    estimation_windows = {
        "1 year": "2008-04-25",
        "3 years": "2006-04-24",
        "5 years": "2004-04-24",
    }

    garch_window_rows = []

    # Keep fitted models and forecasts for later pricing and charts.
    garch_fits = {}
    garch_vol_forecasts = {}

    for window_name, start_date in estimation_windows.items():

        # Select prices before calculating returns.
        window_prices = extended_prices.loc[
            extended_prices["date"].between(
                start_date,
                "2009-04-24"
            )
        ].copy()

        window_returns = np.log(
            window_prices["price"]
            / window_prices["price"].shift(1)
        ).dropna().to_numpy()

        fitted_model = fit_garch(window_returns)
        garch_fits[window_name] = fitted_model

        window_omega = float(fitted_model.params["omega"])
        window_alpha = float(fitted_model.params["alpha[1]"])
        window_beta = float(fitted_model.params["beta[1]"])

        window_persistence = window_alpha + window_beta

        forecast = fitted_model.forecast(
            horizon=252,
            method="analytic",
            reindex=False
        )

        # Convert percentage-squared variance to decimal variance.
        daily_variances = (
            forecast.residual_variance.iloc[-1].to_numpy()
            / 10000
        )

        annualized_vol_forecast = np.sqrt(
            daily_variances * 252
        )

        garch_vol_forecasts[window_name] = (
            annualized_vol_forecast
        )

        # Only calculate a finite long-run variance if supported.
        if window_persistence < 1:
            long_run_vol = np.sqrt(
                (window_omega / (1 - window_persistence))
                / 10000
                * 252
            )
        else:
            long_run_vol = np.nan

        garch_window_rows.append({
            "Window": window_name,
            "First price date": (
                window_prices["date"].iloc[0].date()
            ),
            "Returns": len(window_returns),
            "Historical vol (%)": (
                window_returns.std(ddof=1)
                * np.sqrt(252)
                * 100
            ),
            "Persistence": window_persistence,
            "Next-day vol (%)": (
                annualized_vol_forecast[0] * 100
            ),
            "Day-252 vol (%)": (
                annualized_vol_forecast[-1] * 100
            ),
            "Average-variance vol (%)": (
                np.sqrt(daily_variances.mean() * 252) * 100
            ),
            "Long-run vol (%)": long_run_vol * 100,
        })

    garch_window_comparison = pd.DataFrame(garch_window_rows)

    print("\nGARCH — ESTIMATION WINDOW COMPARISON")
    print("-" * 150)

    print(
        garch_window_comparison.round(4).to_string(index=False)
    )

    print(
        "\nAverage-variance vol = annualized square root "
        "of average forecast daily variance."
    )

    print(
        "A missing long-run volatility means the fitted model "
        "does not have a finite long-run variance."
    )

    garch_window_comparison.to_csv(
        OUTPUT_DIR / "garch_window_comparison.csv",
        index=False
    )

        # Price Asian options under each fitted GARCH window.
    garch_pricing_rows = []
    garch_average_prices_by_window = {}

    for window_name, fitted_model in garch_fits.items():

        garch_averages = simulate_garch_average_prices(
            fitted_model=fitted_model,
            daily_forwards=daily_forwards,
            n_paths=50000,
            seed=42
        )

        garch_average_prices_by_window[window_name] = (
            garch_averages
        )

        # Reuse the same payoff and discounting function.
        results = price_asian_cap_floor(
            average_prices=garch_averages,
            cap_strike=asian_cap_strike,
            floor_strike=asian_floor_strike,
            risk_free_rate=asian_rate,
            discount_years=asian_discount_years
        )

        # Verify forward-curve consistency within simulation noise.
        mean_price = garch_averages.mean()

        mean_se = (
            garch_averages.std(ddof=1)
            / np.sqrt(len(garch_averages))
        )

        baseline_mean = daily_forwards.mean()

        print(f"\nGARCH — {window_name}: SIMULATION CHECK")
        print("-" * 60)

        print(f"Baseline average: ${baseline_mean:.6f}/gal")
        print(f"Simulated mean:   ${mean_price:.6f}/gal")
        print(f"Mean MC SE:       ${mean_se:.6f}/gal")

        print(
            "Difference in standard errors: "
            f"{(mean_price - baseline_mean) / mean_se:.2f}"
        )

        for option, result in results.items():
            garch_pricing_rows.append({
                "Window": window_name,
                "Option": option,
                "KCNB quote": quotes[option],
                "GARCH value": result["value"],
                "MC SE": result["standard_error"],
                "95% MC lower": result["ci_lower"],
                "95% MC upper": result["ci_upper"],
                "Quote minus model": (
                    quotes[option] - result["value"]
                ),
            })

    garch_pricing_comparison = pd.DataFrame(
        garch_pricing_rows
    )

    print("\nASIAN GARCH — PRICE COMPARISON ($/gallon)")
    print("-" * 120)

    print(
        garch_pricing_comparison.round(5).to_string(index=False)
    )

    garch_pricing_comparison.to_csv(
        OUTPUT_DIR / "asian_garch_comparison.csv",
        index=False
    )

    # GARCH cap and floor prices by estimation window.
    window_order = ["1 year", "3 years", "5 years"]

    cap_rows = (
        garch_pricing_comparison.loc[
            garch_pricing_comparison["Option"] == "Cap"
        ]
        .set_index("Window")
        .loc[window_order]
    )

    floor_rows = (
        garch_pricing_comparison.loc[
            garch_pricing_comparison["Option"] == "Floor"
        ]
        .set_index("Window")
        .loc[window_order]
    )

    x = np.arange(len(window_order))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(
        x - 0.10,
        cap_rows["GARCH value"].to_numpy(),
        yerr=1.96 * cap_rows["MC SE"].to_numpy(),
        fmt="o",
        markersize=7,
        capsize=5,
        color="#24476B",
        label="GARCH cap"
    )

    ax.errorbar(
        x + 0.10,
        floor_rows["GARCH value"].to_numpy(),
        yerr=1.96 * floor_rows["MC SE"].to_numpy(),
        fmt="s",
        markersize=7,
        capsize=5,
        color="#C47A28",
        label="GARCH floor"
    )

    ax.axhline(
        asian_cap_quote,
        linestyle="--",
        linewidth=1.8,
        color="#28644A",
        label="KCNB cap quote",
        zorder=2
    )

    ax.axhline(
        asian_floor_quote,
        linestyle=":",
        linewidth=1.8,
        color="#6B2447",
        label="KCNB floor quote",
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(window_order)
    ax.set_xlim(-0.5, len(window_order) - 0.5)
    ax.set_ylim(bottom=0)

    ax.set_title(
        "Asian GARCH Valuation by Estimation Window",
        fontsize=14,
        fontweight="bold",
        pad=14
    )

    ax.set_xlabel("Historical estimation window")
    ax.set_ylabel("Option value ($/gal)")

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.text(
        0.5,
        0.025,
        "Error bars: 95% Monte Carlo intervals; "
        "exclude parameter and model uncertainty.\n"
        "Both KCNB quotes equal $0.079/gal and overlap.",
        ha="center",
        fontsize=9,
        color="#444444"
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    fig.savefig(
        OUTPUT_DIR / "asian_garch_window_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT_DIR / "asian_garch_window_comparison.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    # Load the traded option quotes from Exhibit 4.
    market_options = load_market_options(DATA_DIR)

    print("\nNYMEX MARKET CHECK — INPUTS")
    print("-" * 90)

    print(f"Number of option quotes: {len(market_options)}")

    print(
        market_options[
            [
                "contract",
                "strike",
                "market_premium",
                "futures_price",
                "days_to_expiry",
            ]
        ].head(8).to_string(index=False)
    )

    # Infer volatility from each traded call premium.
    market_iv_results = calculate_market_implied_volatilities(
        market_options
    )

    # Display strikes as rows and maturities as columns.
    contract_order = ["Aug09", "Oct09", "Dec09", "Feb10", "May10"]

    market_iv_table = market_iv_results.pivot(
        index="strike",
        columns="contract",
        values="implied_volatility"
    ).reindex(columns=contract_order)

    print("\nNYMEX — BLACK-76 IMPLIED VOLATILITY (%)")
    print("-" * 75)
    print((market_iv_table * 100).round(2).to_string())

    maximum_repricing_error = (
        market_iv_results["repricing_error"].abs().max()
    )

    print(
        "\nMaximum absolute repricing error: "
        f"${maximum_repricing_error:.12f}/gallon"
    )

    if maximum_repricing_error > 1e-8:
        raise ValueError(
            "Implied-volatility calculation failed to reproduce quotes."
        )

    market_iv_results.to_csv(
        OUTPUT_DIR / "market_implied_volatility_details.csv",
        index=False
    )

    (market_iv_table * 100).to_csv(
        OUTPUT_DIR / "market_implied_volatility_percent.csv"
    )

    # Select the quoted strike closest to each futures price.
    market_plot = market_iv_results.copy()

    market_plot["strike_distance"] = (
        market_plot["strike"] - market_plot["futures_price"]
    ).abs()

    selected_indices = market_plot.groupby(
        "contract"
    )["strike_distance"].idxmin()

    near_atm = (
        market_plot.loc[selected_indices]
        .sort_values("days_to_expiry")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(near_atm))
    market_vol_percent = near_atm["implied_volatility"] * 100

    ax.plot(
        x,
        market_vol_percent,
        color="#24476B",
        marker="o",
        linewidth=2,
        markersize=7,
        label="NYMEX implied volatility: nearest quoted strike"
    )

    ax.axhline(
        annual_vol * 100,
        color="#C47A28",
        linestyle="--",
        linewidth=1.8,
        label=f"Case historical volatility: {annual_vol:.2%}"
    )

    # Approximate range identified by our Asian sensitivity analysis.
    # This is a model-dependent reference, not a confidence interval.
    ax.axhspan(
        38,
        39,
        color="#28644A",
        alpha=0.15,
        label="Asian quote-matching volatility: approx. 38–39%"
    )

    for position, value in zip(x, market_vol_percent):
        ax.annotate(
            f"{value:.2f}%",
            (position, value),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#24476B"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(near_atm["contract"])

    ax.set_xlabel("Futures delivery contract")
    ax.set_ylabel("Annualized volatility (%)")

    ax.set_title(
        "NYMEX Market-Implied Volatility by Maturity",
        fontsize=14,
        fontweight="bold",
        pad=14
    )

    ax.set_ylim(35, max(annual_vol * 100, market_vol_percent.max()) + 5)

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        frameon=False,
        fontsize=9
    )

    fig.text(
        0.5,
        0.02,
        "Source: Exhibit 4. Black-76-equivalent implied volatilities.\n"
        "The shaded range is a constant-volatility Asian-model reference.",
        ha="center",
        fontsize=9,
        color="#444444"
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig.savefig(
        OUTPUT_DIR / "market_implied_volatility.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT_DIR / "market_implied_volatility.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    # Compare every Exhibit 2 quote with our existing simulations.
    kcnb_quotes = pd.read_csv(
        DATA_DIR / "kcnb_quotes.csv"
    ).sort_values("strike").reset_index(drop=True)

    expected_strikes = np.array([
        1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70
    ])

    if (
        len(kcnb_quotes) != len(expected_strikes)
        or not np.allclose(
            kcnb_quotes["strike"].to_numpy(),
            expected_strikes,
            rtol=0.0,
            atol=1e-10
        )
    ):
        raise ValueError("KCNB strikes do not match Exhibit 2.")

    quote_columns = ["strike", "cap_quote", "floor_quote"]

    if not np.isfinite(
        kcnb_quotes[quote_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("KCNB quotes contain invalid values.")

    if (kcnb_quotes[quote_columns] <= 0).any().any():
        raise ValueError("KCNB strikes and premiums must be positive.")

    all_strike_rows = []

    # Reuse the annual averages already generated.
    five_year_garch_averages = (
        garch_average_prices_by_window["5 years"]
    )

    for row in kcnb_quotes.itertuples(index=False):
        strike = row.strike

        # Set both strikes to the same value to price a call
        # and a put separately at this quoted strike.
        bootstrap_values = price_asian_cap_floor(
            average_prices=average_prices,
            cap_strike=strike,
            floor_strike=strike,
            risk_free_rate=asian_rate,
            discount_years=asian_discount_years
        )

        garch_values = price_asian_cap_floor(
            average_prices=five_year_garch_averages,
            cap_strike=strike,
            floor_strike=strike,
            risk_free_rate=asian_rate,
            discount_years=asian_discount_years
        )

        for option, quote in [
            ("Cap", row.cap_quote),
            ("Floor", row.floor_quote),
        ]:
            bootstrap_result = bootstrap_values[option]
            garch_result_at_strike = garch_values[option]

            all_strike_rows.append({
                "Option": option,
                "Strike": strike,
                "KCNB quote": quote,
                "Bootstrap value": bootstrap_result["value"],
                "5Y GARCH value": garch_result_at_strike["value"],
                "Quote minus bootstrap": (
                    quote - bootstrap_result["value"]
                ),
                "Quote minus GARCH": (
                    quote - garch_result_at_strike["value"]
                ),
                "Bootstrap MC SE": (
                    bootstrap_result["standard_error"]
                ),
                "GARCH MC SE": (
                    garch_result_at_strike["standard_error"]
                ),
            })

    all_strike_comparison = pd.DataFrame(all_strike_rows)

    display_columns = [
        "Strike",
        "KCNB quote",
        "Bootstrap value",
        "5Y GARCH value",
        "Quote minus bootstrap",
        "Quote minus GARCH",
    ]

    for option in ["Cap", "Floor"]:
        print(f"\nKCNB — ALL {option.upper()} STRIKES ($/gallon)")
        print("-" * 115)

        option_table = all_strike_comparison.loc[
            all_strike_comparison["Option"] == option,
            display_columns
        ]

        print(option_table.round(5).to_string(index=False))

    all_strike_comparison.to_csv(
        OUTPUT_DIR / "kcnb_all_strikes_comparison.csv",
        index=False
    )
    
    #market parameters for pricing

    future_price = 1.638 # Current futures price ($ per gallon)
    cap_strike = 1.68 # Cap strike price ($ per gallon)
    floor_strike = 1.40 # Floor strike price ($ per gallon)
    risk_free_rate = 0.0049 # Annual risk-free rate (0.49%)
    time_to_expiry = (pd.Timestamp("2010-04-27") - pd.Timestamp("2009-04-24")).days / 365# Time to expiry in years

    # Independent Black-76 benchmark

    cap_model = black76_call(
        future_price,
        cap_strike,
        risk_free_rate,
        time_to_expiry,
        annual_vol
    )

    floor_model = black76_put(
        future_price,
        floor_strike,
        risk_free_rate,
        time_to_expiry,
        annual_vol
    )

    #KCNB quotes (Exhibit 2)
    floor_quote = 0.079 # Floor quote ($ per gallon)
    cap_quote = 0.088 + ((1.68-1.65) / (1.70 - 1.65)) * (0.073 - 0.088) # Cap quote ($ per gallon)

    #Comparison table
    comparison = pd.DataFrame({
        "Option": ["Cap", "Floor"],
        "Strike Price ($/gal)": [cap_strike, floor_strike],
        "KCNB Quote ($/gal)": [cap_quote, floor_quote],
        "Black-76 Model ($/gal)": [cap_model, floor_model]
    })

    comparison["Quote minus Model ($/gal)"] = comparison["KCNB Quote ($/gal)"] - comparison["Black-76 Model ($/gal)"]

    print("\nKCNB price comparison")
    print("-" * 80)
    print(comparison.round(4).to_string(index=False))

    #Exporting the table to CSV
    comparison.to_csv(
        OUTPUT_DIR / "KCNB_price_comparison.csv",
        index=False
    )

    #In case of not trusting the historical volatility, we're going to check the sensitivity of the model to volatility changes
    volatility_scenarios = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60] # Volatility scenarios (30% to 60%)

    sensitivity_rows = []
    for sigma in volatility_scenarios:
        cap_value = black76_call(
            future_price,
            cap_strike,
            risk_free_rate,
            time_to_expiry,
            sigma
        )

        floor_value = black76_put(
            future_price,
            floor_strike,
            risk_free_rate,
            time_to_expiry,
            sigma
        )

        sensitivity_rows.append({
            "Volatility": sigma,
            "Cap Value ($/gal)": cap_value,
            "Floor Value ($/gal)": floor_value,
            "Net Collar Value ($/gal)": cap_value - floor_value,
            "Cap quote minus model ($/gal)": cap_quote - cap_value,
            "Floor quote minus model ($/gal)": floor_quote - floor_value
        })

    sensitivity = pd.DataFrame(sensitivity_rows)

    print("\nVolatility sensitivity analysis")
    print("-" * 100)
    print(sensitivity.round(4).to_string(index=False))
    sensitivity.to_csv(
        OUTPUT_DIR / "volatility_sensitivity.csv",
        index=False
    )

    #Chart of volatility sensitivity
    plt.figure(figsize=(9, 5))
    plt.plot(
        sensitivity["Volatility"] * 100,
        sensitivity["Cap Value ($/gal)"],
        label="Cap Value",
        marker="o",
        color="#24476B"
    )

    plt.plot(
        sensitivity["Volatility"] * 100,
        sensitivity["Floor Value ($/gal)"],
        label="Floor Value",
        marker="o",
        color="#C47A28"
    )

    plt.axhline(
        cap_quote,
        linestyle="--",
        label="KCNB Cap Quote",
        color="#28644A"
    )

    plt.axhline(
        floor_quote,
        linestyle=":",
        label="KCNB Floor Quote",
        color="#6B2447"
    )
    plt.title("Volatility Sensitivity of Option valuation")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Option Value ($/gal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "option_sensitivity.png", dpi=300)
    plt.close()

    print("\nOPTION PRICING")
    print("-" * 45)
    print(f"Historical volatility: {annual_vol:.2%}")
    print(f"cap model value: ${cap_model:.4f} per gallon")
    print(f"floor model value: ${floor_model:.4f} per gallon")
    print(f"Net collar value: ${cap_model - floor_model:.4f} per gallon")

    print(f"Observations: {len(prices)}")
    print(f"First date: {prices['date'].min().date()}")
    print(f"Last date: {prices['date'].max().date()}")
    print(f"Daily volatility: {daily_vol:.2%}")
    print(f"Annualized volatility: {annual_vol:.2%}")

    # Save the processed dataset
    prices.to_csv(
        OUTPUT_DIR / "processed_prices.csv",
        index=False
    )

    # Plot historical prices
    plt.figure(figsize=(10, 5))
    plt.plot(prices["date"], prices["price"], label="Price")
    plt.title("Historical Heating-Oil Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD per gallon)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "historical_prices.png", dpi=300)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
