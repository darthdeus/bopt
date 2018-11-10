import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import solve, cholesky, inv
from numpy.random import multivariate_normal

from scipy.optimize import minimize
from scipy.stats import norm

from functools import partial

import numpy as np
from kernels import sqexp, k


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
    i = 0

    def nll_fn(X_train, y_train, noise, kernel=sqexp):
        def step(theta):
            K = k(X_train, X_train, kernel=partial(sqexp, l=theta[0])) + noise ** 2 * np.eye(len(X_train))

            t1 = 0.5 * y_train @ inv(K) @ y_train
            t2 = 0.5 * np.linalg.det(K)
            t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

            global i
            i += 1
            return t1 + t2 + t3

        return step

    noise = 0.1

    # X_train = np.array([0,1,2,3])
    # y_train = X_train - 1.5 # np.zeros_like(X_train)
    res = minimize(nll_fn(X_train, y_train, noise), [1], bounds=[(1e-5, None)], method="L-BFGS-B")
    l_opt = res.x

    k_opt = partial(sqexp, l=l_opt)

    plot_gp(*gp_reg(X_train, y_train, X, kernel=k_opt), X, X_train, y_train, figure=False)