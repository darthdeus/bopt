from collections import Callable
from typing import NamedTuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import solve, cholesky, inv
from numpy.random import multivariate_normal

from scipy.optimize import minimize
from scipy.stats import norm

from functools import partial

from .plot import plot_gp
from .kernels import Kernel, SquaredExp


class Posterior(NamedTuple):
    mu: np.ndarray
    cov: np.ndarray

    def std(self):
        return np.sqrt(np.diag(self.cov))

    def plot(self):
        pass



class GaussianProcess:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: Optional[np.ndarray]

    mu: np.ndarray
    cov: np.ndarray
    std: np.ndarray

    kernel: Kernel

    K: np.ndarray

    def __init__(self, kernel=SquaredExp()):
        self.kernel = kernel

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, kernel=None) -> "GaussianProcess":
        if kernel is not None:
            self.kernel = kernel

        self.X_train = X_train
        self.y_train = y_train

        self.K = self.kernel(X_train, X_train)

        return self

    # TODO: copy
    def with_kernel(self, kernel):
        self.kernel = kernel
        return self

    def with_kernel_params(self, theta):
        self.kernel = self.kernel.with_params(theta)

        return self

    def posterior(self, X_test: np.ndarray, X_train: np.ndarray = None, y_train: np.ndarray = None):
        """
        Computes the posterior `p(y_test | X_test, X_train, y_train)`
        and stores the result.
        """
        if (X_train is not None) and (y_train is not None):
            self.fit(X_train, y_train)

        self.X_test = X_test
        n = len(X_test)

        K = self.K
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        stable_eye = 1e-4 * np.eye(len(K))  # Just for numerical stability?

        K_inv = inv(K + stable_eye)

        # L = cholesky(K + stable_eye)
        # alpha = solve(L.T, solve(L, self.y_train))
        # mu = K_s.T @ alpha

        self.mu = K_s.T @ K_inv @ self.y_train
        self.cov = K_ss - K_s.T @ K_inv @ K_s
        self.std = np.sqrt(np.diag(self.cov))

        return self

    def optimize_kernel(self, X_train, y_train):
        noise_level = 0.1

        def step(theta):
            noise = noise_level ** 2 * np.eye(len(X_train))
            K = self.kernel.with_params(theta)(X_train, X_train) + noise

            t1 = 0.5 * y_train @ inv(K) @ y_train
            t2 = 0.5 * np.linalg.det(K)
            t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

            return t1 + t2 + t3

        res = minimize(step,
                       self.kernel.default_params(),
                       bounds=self.kernel.param_bounds(), method="L-BFGS-B")

        self.kernel = self.kernel.with_params(res.x)

        return self

    def plot_prior(self, X, **kwargs):
        plot_gp(np.zeros(len(X)), self.kernel(X, X), X, **kwargs)

    def plot_posterior(self, **kwargs):
        plot_gp(self.mu, self.cov, self.X_test, self.X_train, self.y_train, **kwargs)


# gp = GaussianProcess().fit(X_train, y_train)
# mu, cov = gp.posterior(X_test)

# gp.plot_prior()
# gp.plot_posterior()

#
# def gp_reg(X_train, y_train, X_test, kernel=sqexp, return_std=False):
#     n = len(X_test)
#
#     K = k(X_train, X_train, kernel)
#     K_s = k(X_train, X_test, kernel)
#     K_ss = k(X_test, X_test, kernel)
#
#     stable_eye = 1e-4 * np.eye(len(K))  # Just for numerical stability?
#
#     K_inv = inv(K + stable_eye)
#
#     # L = cholesky(K + stable_eye)
#     # alpha = solve(L.T, solve(L, y_train))
#     # mu = K_s.T @ alpha
#
#     mu = K_s.T @ K_inv @ y_train
#     cov = K_ss - K_s.T @ K_inv @ K_s
#
#     if return_std:
#         return mu, np.sqrt(np.diag(cov))
#     else:
#         return mu, cov
#
#
