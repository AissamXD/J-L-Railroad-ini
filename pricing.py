import numpy as np
from scipy.stats import norm


def black76_call(F, K, r, T, sigma):
    """
    Price a European call option on a futures contract
    using the Black-76 model.

    F: futures price ($ per gallon)
    K: strike price ($ per gallon)
    r: annual continuously compounded risk-free rate
    T: time to expiry in years
    sigma: annualized volatility (e.g. 0.40 for 40%)
    """

    if sigma < 0 or T < 0:
        raise ValueError("Volatility and time must be non-negative.")

    if T == 0:
        return max(F - K, 0)

    if sigma == 0:
        return np.exp(-r * T) * max(F - K, 0)

    d1 = (
        np.log(F / K) + 0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call_value = np.exp(-r * T) * (
        F * norm.cdf(d1) - K * norm.cdf(d2)
    )

    return call_value


def black76_put(F, K, r, T, sigma):
    """
    Price a European put option on a futures contract
    using the Black-76 model.
    """

    if sigma < 0 or T < 0:
        raise ValueError("Volatility and time must be non-negative.")

    if T == 0:
        return max(K - F, 0)

    if sigma == 0:
        return np.exp(-r * T) * max(K - F, 0)

    d1 = (
        np.log(F / K) + 0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    put_value = np.exp(-r * T) * (
        K * norm.cdf(-d2) - F * norm.cdf(-d1)
    )

    return put_value