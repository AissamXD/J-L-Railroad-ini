import numpy as np

def prepare_bootstrap_returns(log_returns):
    """
    Prepare historical daily log returns for independent resampling.

    Adjust the returns so that the empirical average of their
    gross multipliers, exp(adjusted_return), equals one.

    This prevents historical drift from moving the expected
    simulated price away from the forward-curve baseline.
    """
    returns = np.asarray(log_returns, dtype=float)
    
    if returns.ndim != 1:
        raise ValueError("log_returns must be a 1D array or list.")

    if len(returns) < 2:
        raise ValueError("log_returns must contain at least two values.")

    if not np.isfinite(returns).all():
        raise ValueError("log_returns contains non-finite values.")

    #Convert log returns into daily price multipliers
    gross_multipliers = np.exp(returns)

    #Find the average historical multiplier
    mean_gross_multiplier = gross_multipliers.mean()

    #Adjust log returns so the average adjusted multiplier equals one
    drift_adjustment = np.log(mean_gross_multiplier)
    adjusted_returns = returns - drift_adjustment

    return adjusted_returns, drift_adjustment


def build_daily_forward_curve(forward_curve, n_steps=252):
    """
    Interpolate monthly futures prices into 252 model observations.

    Monthly quotes are anchored at their first-of-month labels.
    The window runs from May 1, 2009 through April 30, 2010.
    Observations approximate trading days, not an exact calendar.
    """

    curve = forward_curve.sort_values(
        "delivery_month"
    ).reset_index(drop=True)

    start_date = curve["delivery_month"].iloc[0]

    # Convert monthly dates into numbers of days from May 1, 2009.
    monthly_offsets = (
        curve["delivery_month"] - start_date
    ).dt.days.to_numpy(dtype=float)

    monthly_prices = curve["futures_price"].to_numpy(dtype=float)

    # End on April 30, one day before the May 2010 anchor.
    last_observation = monthly_offsets[-1] - 1

    observation_offsets = np.linspace(
        0,
        last_observation,
        n_steps
    )

    # Connect the monthly quotes with straight lines.
    daily_forwards = np.interp(
        observation_offsets,
        monthly_offsets,
        monthly_prices
    )

    return daily_forwards

def simulate_bootstrap_paths(
    daily_forwards,
    bootstrap_returns,
    n_paths=10000,
    seed=42
):
    """
    Simulate Asian-option price paths using independently
    resampled, drift-adjusted historical log returns.

    Returns an array with:
        rows = simulated paths
        columns = daily observations
    """

    daily_forwards = np.asarray(daily_forwards, dtype=float)
    bootstrap_returns = np.asarray(bootstrap_returns, dtype=float)

    if daily_forwards.ndim != 1 or bootstrap_returns.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional.")

    if len(daily_forwards) == 0 or len(bootstrap_returns) == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.isfinite(daily_forwards).all():
        raise ValueError("Forward prices must be finite.")

    if (daily_forwards <= 0).any():
        raise ValueError("Forward prices must be positive.")

    if not np.isfinite(bootstrap_returns).all():
        raise ValueError("Bootstrap returns must be finite.")

    if n_paths < 2:
        raise ValueError("At least two simulated paths are required.")

    n_steps = len(daily_forwards)

    # A fixed seed makes the simulation reproducible.
    rng = np.random.default_rng(seed)

    # Draw one historical shock for each observation in each path.
    sampled_returns = rng.choice(
        bootstrap_returns,
        size=(n_paths, n_steps),
        replace=True
    )

    # Accumulate log returns along each path.
    cumulative_returns = np.cumsum(
        sampled_returns,
        axis=1
    )

    # Convert cumulative log returns into price multipliers.
    price_multipliers = np.exp(cumulative_returns)

    # Apply the multipliers to the forward baseline.
    simulated_prices = (
        daily_forwards[np.newaxis, :] * price_multipliers
    )

    return simulated_prices

def price_asian_cap_floor(
    average_prices,
    cap_strike,
    floor_strike,
    risk_free_rate,
    discount_years
):
    """
    Calculate discounted Asian cap, floor and net collar values.

    Standard errors measure Monte Carlo sampling uncertainty,
    conditional on the model and its inputs.
    """

    average_prices = np.asarray(average_prices, dtype=float)

    if average_prices.ndim != 1 or len(average_prices) < 2:
        raise ValueError("Provide at least two simulated average prices.")

    if not np.isfinite(average_prices).all():
        raise ValueError("Simulated average prices must be finite.")

    if cap_strike <= 0 or floor_strike <= 0:
        raise ValueError("Strikes must be positive.")

    if discount_years < 0:
        raise ValueError("Discounting time cannot be negative.")

    discount_factor = np.exp(
        -risk_free_rate * discount_years
    )

    # Payoffs per gallon for each simulated annual average.
    cap_payoffs = np.maximum(
        average_prices - cap_strike,
        0.0
    )

    floor_payoffs = np.maximum(
        floor_strike - average_prices,
        0.0
    )

    discounted_cap = discount_factor * cap_payoffs
    discounted_floor = discount_factor * floor_payoffs

    # J&L buys the cap and sells the floor.
    discounted_collar = discounted_cap - discounted_floor

    results = {}

    for name, values in [
        ("Cap", discounted_cap),
        ("Floor", discounted_floor),
        ("Net collar", discounted_collar),
    ]:
        estimate = values.mean()
        standard_error = (
            values.std(ddof=1) / np.sqrt(len(values))
        )

        results[name] = {
            "value": estimate,
            "standard_error": standard_error,
            "ci_lower": estimate - 1.96 * standard_error,
            "ci_upper": estimate + 1.96 * standard_error,
        }

    return results

def scale_returns_to_volatility(
    historical_returns,
    target_annual_volatility,
    trading_days=252
):
    """
    Rescale historical log returns to a target annual volatility,
    then adjust their drift to preserve the forward baseline.
    """

    returns = np.asarray(historical_returns, dtype=float)

    if returns.ndim != 1 or len(returns) < 2:
        raise ValueError("Provide at least two historical returns.")

    if not np.isfinite(returns).all():
        raise ValueError("Historical returns must be finite.")

    if target_annual_volatility < 0:
        raise ValueError("Target volatility cannot be negative.")

    # Remove the historical average log return.
    centered_returns = returns - returns.mean()

    historical_annual_volatility = (
        centered_returns.std(ddof=1) * np.sqrt(trading_days)
    )

    if historical_annual_volatility <= 0:
        raise ValueError("Historical volatility must be positive.")

    # Change the size of shocks while preserving their shape.
    scaled_returns = centered_returns * (
        target_annual_volatility / historical_annual_volatility
    )

    # Restore an average gross multiplier of one after scaling.
    adjusted_returns, _ = prepare_bootstrap_returns(
        scaled_returns
    )

    return adjusted_returns