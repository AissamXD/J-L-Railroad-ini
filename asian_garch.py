import numpy as np
from arch import arch_model


def fit_garch(historical_returns):
    """
    Fit a constant-mean GARCH(1,1) model to historical log returns.

    Input returns are decimals.
    Estimation uses percentage returns for numerical stability.
    """

    returns = np.asarray(historical_returns, dtype=float)

    if returns.ndim != 1 or len(returns) < 2:
        raise ValueError("Provide a one-dimensional return series.")

    if not np.isfinite(returns).all():
        raise ValueError("Historical returns must be finite.")

    # Example: a decimal return of 0.02 becomes 2.0.
    returns_pct = returns * 100

    model = arch_model(
        returns_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False
    )

    result = model.fit(
        disp="off",
        update_freq=0
    )

    if result.convergence_flag != 0:
        raise RuntimeError(
            "GARCH estimation did not converge. "
            "Review the optimizer message before continuing."
        )

    return result

def simulate_garch_average_prices(
    fitted_model,
    daily_forwards,
    n_paths=50000,
    seed=42
):
    """
    Simulate annual-average prices with Gaussian GARCH(1,1)
    variance dynamics and a forward-curve-consistent price drift.

    Pricing assumption:
    Historically fitted variance parameters are retained.
    No separate volatility-risk-premium adjustment is estimated.
    """

    daily_forwards = np.asarray(daily_forwards, dtype=float)

    if daily_forwards.ndim != 1 or len(daily_forwards) == 0:
        raise ValueError("Provide a one-dimensional forward curve.")

    if not np.isfinite(daily_forwards).all():
        raise ValueError("Forward prices must be finite.")

    if (daily_forwards <= 0).any():
        raise ValueError("Forward prices must be positive.")

    if n_paths < 2:
        raise ValueError("At least two paths are required.")

    # The fitted model used returns in percentage units.
    # Convert the variance intercept to decimal-return units.
    omega = float(fitted_model.params["omega"]) / 10000
    alpha = float(fitted_model.params["alpha[1]"])
    beta = float(fitted_model.params["beta[1]"])

    # Start from the fitted next-day conditional variance.
    initial_forecast = fitted_model.forecast(
        horizon=1,
        method="analytic",
        reindex=False
    )

    initial_variance = float(
        initial_forecast.residual_variance.iloc[-1, 0]
    ) / 10000

    rng = np.random.default_rng(seed)

    # Each path develops its own variance and price multiplier.
    daily_variance = np.full(n_paths, initial_variance)
    price_multiplier = np.ones(n_paths)
    price_sum = np.zeros(n_paths)

    for forward_price in daily_forwards:

        # Independent standard-normal shocks across paths.
        z = rng.standard_normal(n_paths)

        shock = np.sqrt(daily_variance) * z

        # Conditional mean of this gross multiplier is exactly 1
        # under the normal-shock assumption.
        price_multiplier *= np.exp(
            shock - 0.5 * daily_variance
        )

        simulated_price = forward_price * price_multiplier
        price_sum += simulated_price

        # Update next day's variance separately for every path.
        daily_variance = (
            omega
            + alpha * shock**2
            + beta * daily_variance
        )

    average_prices = price_sum / len(daily_forwards)

    if not np.isfinite(average_prices).all():
        raise ValueError(
            "Simulation produced non-finite prices. "
            "Inspect the fitted parameters."
        )

    return average_prices