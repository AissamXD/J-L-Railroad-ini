import numpy as np
import pandas as pd
from scipy.optimize import brentq
from pricing import black76_call

def load_market_options(data_dir):
    """Load and combine the option premiums and terms in Exhibit 4."""

    call_prices = pd.read_csv(
        data_dir / "market_call_prices.csv"
    )

    terms = pd.read_csv(
        data_dir / "market_option_terms.csv",
        parse_dates=["expiry_date"]
    )

    if call_prices["strike"].duplicated().any():
        raise ValueError("Duplicate strikes in market_call_prices.csv")

    if terms["contract"].duplicated().any():
        raise ValueError("Duplicate contracts in market_option_terms.csv")

    # Turn the wide table into one row per option.
    options = call_prices.melt(
        id_vars="strike",
        var_name="contract",
        value_name="market_premium"
    )

    # Attach the appropriate futures price, rate and expiry.
    options = options.merge(
        terms,
        on="contract",
        how="left",
        validate="many_to_one"
    )

    if options.isna().any().any():
        raise ValueError("Missing premiums or unmatched contract terms.")

    numeric_columns = [
        "strike",
        "market_premium",
        "days_to_expiry",
        "futures_price",
        "risk_free_rate",
    ]

    if not np.isfinite(
        options[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Market inputs must be finite numbers.")

    positive_columns = [
        "strike",
        "market_premium",
        "days_to_expiry",
        "futures_price",
    ]

    if (options[positive_columns] <= 0).any().any():
        raise ValueError("Prices, strikes and expiry days must be positive.")

    # Verify the stated day counts against the market-data date.
    calculated_days = (
        options["expiry_date"] - pd.Timestamp("2009-04-24")
    ).dt.days

    # Preserve Exhibit 4's stated counts for transparency.
    options["stated_days_to_expiry"] = options["days_to_expiry"]

    # Use actual calendar differences from April 24, 2009.
    options["days_to_expiry"] = calculated_days
    options["time_to_expiry"] = calculated_days / 365

    return options.sort_values(
        ["days_to_expiry", "strike"]
    ).reset_index(drop=True)

def calculate_market_implied_volatilities(market_options):
    """
    Calculate Black-76-equivalent implied volatility
    for each observed NYMEX call premium.
    """

    results = market_options.copy()

    implied_volatilities = []
    repriced_premiums = []

    for row in results.itertuples(index=False):
        F = row.futures_price
        K = row.strike
        r = row.risk_free_rate
        T = row.time_to_expiry
        premium = row.market_premium

        if T <= 0:
            raise ValueError("Option expiry must be after valuation.")

        discount_factor = np.exp(-r * T)

        minimum_price = discount_factor * max(F - K, 0)
        maximum_price = discount_factor * F

        if not minimum_price <= premium < maximum_price:
            raise ValueError(
                f"{row.contract}, strike {K:.2f}: "
                "premium is outside Black-76 price bounds."
            )

        if premium == minimum_price:
            implied_vol = 0.0
        else:
            def pricing_error(sigma):
                return black76_call(F, K, r, T, sigma) - premium

            if pricing_error(5.0) < 0:
                raise ValueError(
                    f"{row.contract}, strike {K:.2f}: "
                    "implied volatility exceeds the 500% search limit."
                )

            implied_vol = brentq(
                pricing_error,
                0.0,
                5.0,
                xtol=1e-12
            )

        repriced = black76_call(F, K, r, T, implied_vol)

        implied_volatilities.append(implied_vol)
        repriced_premiums.append(repriced)

    results["implied_volatility"] = implied_volatilities
    results["repriced_premium"] = repriced_premiums

    results["repricing_error"] = (
        results["repriced_premium"] - results["market_premium"]
    )

    return results