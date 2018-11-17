import numpy as np
from scipy.stats import norm

from .gaussian_process import GaussianProcess


def expected_improvement(gp: GaussianProcess, X: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
    mu, sigma = gp.posterior(X).mu_std()

    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    improvement = mu - f_s - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    return ei
