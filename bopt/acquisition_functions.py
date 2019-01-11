from typing import Callable

import numpy as np
from scipy.stats import norm

from bopt.gaussian_process import GaussianProcess


AcquisitionFunction = Callable[[GaussianProcess, np.ndarray, float], np.ndarray]


def expected_improvement(gp: GaussianProcess, X: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
    assert X is not None

    mu, sigma = gp.posterior(X).mu_std()

    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1) + 1e-6

    improvement = mu - f_s - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    return ei


def probability_of_improvement(gp: GaussianProcess, X: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
    assert X is not None

    mu, sigma = gp.posterior(X).mu_std()

    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1) + 1e-6

    improvement = mu - f_s - xi
    Z = improvement / sigma

    return norm.cdf(Z)


def get_acquisition_fn_by_name(name):
    mapping = {
            "ei": expected_improvement,
            "pi": probability_of_improvement
            }

    return mapping[name]
