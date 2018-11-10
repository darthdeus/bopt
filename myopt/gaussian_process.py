from collections import Callable
from typing import NamedTuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import solve, cholesky, inv
from numpy.random import multivariate_normal

from scipy.optimize import minimize
from scipy.stats import norm

from functools import partial

import numpy as np

from .plot import plot_gp
from .kernels import sqexp, k


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

    mu: Optional[np.ndarray]
    cov: Optional[np.ndarray]

    kernel: any #: Callable[[np.ndarray, np.ndarray], np.ndarray]

    K: np.ndarray

    def __init__(self, kernel=sqexp):
        self.kernel = kernel

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, kernel=None) -> "GaussianProcess":
        if kernel is not None:
            self.kernel = kernel

        self.X_train = X_train
        self.y_train = y_train

        self.K = k(X_train, X_train, self.kernel)

        return self

    def posterior(self, X_test, return_std=False):
        """
        p(y_test | X_test, X_train, y_train)
        """
        self.X_test = X_test
        n = len(X_test)

        K = self.K
        K_s = k(self.X_train, X_test, self.kernel)
        K_ss = k(X_test, X_test, self.kernel)

        stable_eye = 1e-4 * np.eye(len(K))  # Just for numerical stability?

        K_inv = inv(K + stable_eye)

        # L = cholesky(K + stable_eye)
        # alpha = solve(L.T, solve(L, y_train))
        # mu = K_s.T @ alpha

        self.mu = K_s.T @ K_inv @ self.y_train
        self.cov = K_ss - K_s.T @ K_inv @ K_s

        return self

        # if return_std:
        #     return mu, np.sqrt(np.diag(cov))
        # else:
        #     return mu, cov

    def plot_prior(self, X, num_samples=3):
        # plot_gp(np.zeros(len(X)), k(X, X), X)
        plot_gp(np.zeros(len(X)), k(X, X), X) # , num_samples=num_samples)

    def plot_posterior(self, num_samples=3):
        plot_gp(self.mu, self.cov, self.X_test, self.X_train, self.y_train, num_samples=num_samples)



# gp = GaussianProcess().fit(X_train, y_train)
# mu, cov = gp.posterior(X_test)

# gp.plot_prior()
# gp.plot_posterior()



def gp_reg(X_train, y_train, X_test, kernel=sqexp, return_std=False):
    n = len(X_test)

    K = k(X_train, X_train, kernel)
    K_s = k(X_train, X_test, kernel)
    K_ss = k(X_test, X_test, kernel)

    stable_eye = 1e-4 * np.eye(len(K))  # Just for numerical stability?

    K_inv = inv(K + stable_eye)

    # L = cholesky(K + stable_eye)
    # alpha = solve(L.T, solve(L, y_train))
    # mu = K_s.T @ alpha

    mu = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s

    if return_std:
        return mu, np.sqrt(np.diag(cov))
    else:
        return mu, cov


def optimize_kernel(X_train, y_train, X):
    def nll_fn(X_train, y_train, noise, kernel=sqexp):
        def step(theta):
            K = k(X_train, X_train, kernel=partial(sqexp, l=theta[0])) + noise ** 2 * np.eye(len(X_train))

            t1 = 0.5 * y_train @ inv(K) @ y_train
            t2 = 0.5 * np.linalg.det(K)
            t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

            return t1 + t2 + t3

        return step

    noise = 0.1

    # X_train = np.array([0,1,2,3])
    # y_train = X_train - 1.5 # np.zeros_like(X_train)
    res = minimize(nll_fn(X_train, y_train, noise), [1], bounds=[(1e-5, None)], method="L-BFGS-B")
    l_opt = res.x

    k_opt = partial(sqexp, l=l_opt)

    return k_opt