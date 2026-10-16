"""
Monte Carlo option pricing with variance reduction.

Prices a European call and an arithmetic-average Asian call under
geometric Brownian motion, and compares three estimators:

    plain       - standard Monte Carlo
    antithetic  - pairs each random draw Z with -Z
    control     - control variate (see below)

European call control variate: the discounted terminal price exp(-rT) * S_T
has known expectation S0, so we subtract off its (scaled) noise.

Asian call control variate: the geometric-average Asian call has a closed-form
price and is very highly correlated with the arithmetic one, so it removes
most of the variance.

Run:  python mc_options.py
"""
import argparse
import math

import numpy as np


# ----------------------------------------------------------------------------
# Closed-form prices (used as ground truth / control variates)
# ----------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def geometric_asian_call(S0, K, r, sigma, T, steps):
    """Discretely monitored geometric-average Asian call (closed form)."""
    n = steps
    mu_g = math.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var_g = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd_g = math.sqrt(var_g)
    d1 = (mu_g + var_g - math.log(K)) / sd_g
    d2 = d1 - sd_g
    return math.exp(-r * T) * (math.exp(mu_g + 0.5 * var_g) * norm_cdf(d1) - K * norm_cdf(d2))


# ----------------------------------------------------------------------------
# Path simulation
# ----------------------------------------------------------------------------
def simulate_paths(Z, S0, r, sigma, T):
    """Exact GBM paths from standard normals Z of shape (paths, steps)."""
    steps = Z.shape[1]
    dt = T / steps
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    return S0 * np.exp(np.cumsum(log_increments, axis=1))


def _summarise(samples):
    """Mean and standard error of i.i.d. samples."""
    return samples.mean(), samples.std(ddof=1) / math.sqrt(len(samples))


def _control_adjust(y, x, x_mean):
    """Control variate estimator: y - beta * (x - E[x]), beta = Cov(y,x)/Var(x)."""
    beta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return y - beta * (x - x_mean)


# ----------------------------------------------------------------------------
# European call
# ----------------------------------------------------------------------------
def mc_european_call(n, S0, K, r, sigma, T, method="plain", seed=0):
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T)

    if method == "antithetic":
        half = rng.standard_normal(n // 2)
        Z = np.concatenate([half, -half])
        # average each pair so the n/2 pair-averages are independent samples
        payoff = disc * np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z) - K, 0)
        samples = 0.5 * (payoff[: n // 2] + payoff[n // 2 :])
        return _summarise(samples)

    Z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    y = disc * np.maximum(ST - K, 0)

    if method == "control":
        y = _control_adjust(y, disc * ST, S0)  # E[exp(-rT) S_T] = S0
    return _summarise(y)


# ----------------------------------------------------------------------------
# Arithmetic Asian call
# ----------------------------------------------------------------------------
def mc_asian_call(n, S0, K, r, sigma, T, steps=50, method="plain", seed=0):
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T)

    def payoffs(Z):
        paths = simulate_paths(Z, S0, r, sigma, T)
        arith = disc * np.maximum(paths.mean(axis=1) - K, 0)
        geo = disc * np.maximum(np.exp(np.log(paths).mean(axis=1)) - K, 0)
        return arith, geo

    if method == "antithetic":
        half = rng.standard_normal((n // 2, steps))
        a1, _ = payoffs(half)
        a2, _ = payoffs(-half)
        return _summarise(0.5 * (a1 + a2))

    Z = rng.standard_normal((n, steps))
    arith, geo = payoffs(Z)
    if method == "control":
        arith = _control_adjust(arith, geo, geometric_asian_call(S0, K, r, sigma, T, steps))
    return _summarise(arith)


# ----------------------------------------------------------------------------
# Self-checks and demo
# ----------------------------------------------------------------------------
def self_check(params):
    """Every estimator must land within 4 standard errors of the exact price."""
    exact = black_scholes_call(**params)
    for method in ("plain", "antithetic", "control"):
        est, se = mc_european_call(200_000, method=method, **params)
        assert abs(est - exact) < 4 * se, f"{method} is off: {est:.4f} vs {exact:.4f}"
    print("Self-check passed: all estimators agree with Black-Scholes.\n")


def convergence_plot(params, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = black_scholes_call(**params)
    ns = np.unique(np.logspace(3, 6, 14).astype(int) // 2 * 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, colour in (("plain", "#c0392b"), ("antithetic", "#e67e22"), ("control", "#1f3a5f")):
        errs = [np.mean([abs(mc_european_call(int(n), method=method, seed=s, **params)[0] - exact)
                         for s in range(20)]) for n in ns]
        ax.loglog(ns, errs, "o-", ms=4, label=method, color=colour)
    ax.loglog(ns, 10.0 / np.sqrt(ns), "k--", lw=1, label=r"slope $-1/2$ (the $1/\sqrt{N}$ rate)")
    ax.set_xlabel("Number of simulated paths N")
    ax.set_ylabel("Mean absolute error vs Black-Scholes")
    ax.set_title("Monte Carlo convergence and variance reduction (European call)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved convergence plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", type=int, default=200_000)
    parser.add_argument("--plot", default="convergence.png")
    args = parser.parse_args()

    params = dict(S0=100.0, K=105.0, r=0.03, sigma=0.2, T=1.0)
    print(f"Parameters: {params}\n")
    self_check(params)

    exact = black_scholes_call(**params)
    print(f"European call  (exact Black-Scholes = {exact:.4f}, N = {args.paths:,})")
    print(f"{'method':<12}{'estimate':>10}{'std error':>12}{'variance reduction':>22}")
    base_se = None
    for method in ("plain", "antithetic", "control"):
        est, se = mc_european_call(args.paths, method=method, **params)
        base_se = base_se or se
        print(f"{method:<12}{est:>10.4f}{se:>12.5f}{(base_se / se) ** 2:>21.1f}x")

    print(f"\nArithmetic Asian call, 50 monitoring dates (no closed form, N = {args.paths:,})")
    print(f"{'method':<12}{'estimate':>10}{'std error':>12}{'variance reduction':>22}")
    base_se = None
    for method in ("plain", "antithetic", "control"):
        est, se = mc_asian_call(args.paths, method=method, steps=50, **params)
        base_se = base_se or se
        print(f"{method:<12}{est:>10.4f}{se:>12.5f}{(base_se / se) ** 2:>21.1f}x")

    print()
    convergence_plot(params, args.plot)


if __name__ == "__main__":
    main()
