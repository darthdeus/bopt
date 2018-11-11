from collections import Callable
from typing import NamedTuple, Optional, Tuple

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
    noise: float

    kernel: Kernel

    K: np.ndarray

    def __init__(self, noise=0, kernel=SquaredExp()):
        self.noise = noise
        self.kernel = kernel

        self.X_train = None
        self.y_train = None
        self.X_test = None

        self.mu = None
        self.cov = None
        self.std = None

        self.K = None

    def refit(self):
        if self.X_train is not None and self.y_train is not None:
            noise = self.noise * np.eye(len(self.X_train))

            self.K = self.kernel(self.X_train, self.X_train) + noise

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, kernel=None) -> "GaussianProcess":
        if kernel is not None:
            self.kernel = kernel

        self.X_train = X_train
        self.y_train = y_train

        self.refit()

        return self

    def posterior(self, X_test: np.ndarray, X_train: np.ndarray = None, y_train: np.ndarray = None):
        """
        Computes the posterior `p(y_test | X_test, X_train, y_train)`
        and stores the result.
        """
        if (X_train is not None) and (y_train is not None):
            self.fit(X_train, y_train)

        assert self.X_train is not None, "X_train is None, call `fit` first, or provide X_train directly"
        assert self.y_train is not None, "y_train is None, call `fit` first, or provide y_train directly"

        self.X_test = X_test

        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        stable_eye = 1e-4 * np.eye(len(self.K))  # Just for numerical stability?

        K_inv = inv(self.K + stable_eye)

        # L = cholesky(K + stable_eye)
        # alpha = solve(L.T, solve(L, self.y_train))
        # mu = K_s.T @ alpha

        self.mu = K_s.T @ K_inv @ self.y_train
        self.cov = K_ss - K_s.T @ K_inv @ K_s
        self.std = np.sqrt(np.diag(self.cov))

        return self

    def optimize_kernel(self):
        noise_level = 0.1

        assert self.X_train is not None, "X_train is None, call `fit` first"
        assert self.y_train is not None, "y_train is None, call `fit` first"

        def step(theta):
            noise = noise_level ** 2 * np.eye(len(self.X_train))
            K = self.kernel.with_params(theta)(self.X_train, self.X_train) + noise

            t1 = 0.5 * self.y_train @ inv(K) @ self.y_train
            t2 = 0.5 * np.linalg.det(K)
            t3 = 0.5 * len(self.X_train) * np.log(2 * np.pi)

            return t1 + t2 + t3

        default_params = self.kernel.default_params()

        if len(default_params) == 0:
            return self

        res = minimize(step,
                       default_params,
                       bounds=self.kernel.param_bounds(), method="L-BFGS-B")

        self.kernel = self.kernel.with_params(res.x)
        self.refit()

        return self

    def plot_prior(self, X, **kwargs):
        plot_gp(np.zeros(len(X)), self.kernel(X, X), X, **kwargs)

    def plot_posterior(self, **kwargs):
        assert self.X_test is not None, "X_test was not provided, call `.posterior(X_test)` first"
        plot_gp(self.mu, self.cov, self.X_test, self.X_train, self.y_train, **kwargs)

    def copy(self) -> "GaussianProcess":
        gp = GaussianProcess()
        gp.kernel = self.kernel
        gp.noise = self.noise

        gp.X_train = self.X_train
        gp.y_train = self.y_train
        gp.X_test = self.X_test

        gp.mu = self.mu
        gp.cov = self.cov
        gp.std = self.std

        gp.K = self.K

        return gp

    def with_kernel(self, kernel: Kernel) -> "GaussianProcess":
        gp = self.copy()
        gp.kernel = kernel

        gp.refit()

        return gp

    def with_kernel_params(self, theta) -> "GaussianProcess":
        gp = self.copy()
        gp.kernel = self.kernel.with_params(theta)

        gp.refit()

        return gp

    def with_noise(self, noise: float) -> "GaussianProcess":
        gp = self.copy()
        gp.noise = noise

        gp.refit()

        return gp

    def mu_std(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mu, self.std