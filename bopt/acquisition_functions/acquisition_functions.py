import abc
import logging

from GPy.models import GPRegression

import numpy as np
from scipy.stats import norm


# AcquisitionFunction = Callable[[GPRegression, np.ndarray, float], np.ndarray]


class AcquisitionFunction(abc.ABC):
    # TODO: xi default value 0.01 or tweak it?
    def __call__(self, gp: GPRegression, X: np.ndarray, f_s: float, xi: float = 0.01) -> np.ndarray:
        assert X is not None

        mu, var = gp.predict(X)
        if var <= 0:
            logging.debug("""
GPy predicted negative variance, setting to 0 instead. This is a known issue.

See https://github.com/SheffieldML/GPy/issues/253 for more details""")
            var = 1e-6

        sigma = np.sqrt(var)

        ei = self.raw_call(mu, sigma, f_s, xi)

        return ei

    @abc.abstractmethod
    def raw_call(self, mu: np.ndarray, sigma: np.ndarray, f_s: float, xi: float = 0.01) -> np.ndarray:
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass


class ExpectedImprovement(AcquisitionFunction):
    def raw_call(self, mu: np.ndarray, sigma: np.ndarray, f_s: float, xi: float = 0.01) -> np.ndarray:

        improvement = mu - f_s - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    def name(self) -> str:
        return "ei"


class ProbabilityOfImprovement(AcquisitionFunction):
    def raw_call(self, mu: np.ndarray, sigma: np.ndarray, f_s: float, xi: float = 0.01) -> np.ndarray:
        improvement = mu - f_s - xi
        Z = improvement / sigma

        return norm.cdf(Z)

    def name(self) -> str:
        return "pi"


# def expected_improvement(gp: GPRegression, X: np.ndarray, f_s: float, xi:
#         float=0.001) -> np.ndarray:
#     assert X is not None
#
#     mu, sigma = gp.predict(X)
#
#     return expected_improvement_f(mu, sigma, f_s, xi)
#
#
# def expected_improvement_f(mu: np.ndarray, sigma: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
#     # TODO: neni to opacne?
#     improvement = mu - f_s - xi
#     Z = improvement / sigma
#     ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
#
#     return ei


# def probability_of_improvement(gp: GPRegression, X: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
#     assert X is not None
#
#     mu, sigma = gp.predict(X)
#
#     improvement = mu - f_s - xi
#     Z = improvement / sigma
#
#     return norm.cdf(Z)
#
#
# def get_acquisition_fn_by_name(name):
#     mapping = {
#         "ei": expected_improvement,
#         "pi": probability_of_improvement
#     }
#
#     return mapping[name]
