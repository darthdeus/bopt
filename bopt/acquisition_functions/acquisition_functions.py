from typing import Callable
from GPy.models import GPRegression

import numpy as np
from scipy.stats import norm


AcquisitionFunction = Callable[[GPRegression, np.ndarray, float], np.ndarray]


def expected_improvement(gp: GPRegression, X: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
    assert X is not None

    mu, sigma = gp.predict(X)

    improvement = mu - f_s - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    return ei


def probability_of_improvement(gp: GPRegression, X: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
    assert X is not None

    mu, sigma = gp.predict(X)

    improvement = mu - f_s - xi
    Z = improvement / sigma

    return norm.cdf(Z)


def get_acquisition_fn_by_name(name):
    mapping = {
        "ei": expected_improvement,
        "pi": probability_of_improvement
    }

    return mapping[name]
