from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, cholesky, det, solve
from scipy.optimize import minimize

from bopt.kernels import Kernel


def kernel_log_likelihood(kernel: Kernel, X_train: np.ndarray,
                          y_train: np.ndarray, noise_level: float = 0) -> float:
    noise = noise_level ** 2 * np.eye(len(X_train))
    K = kernel(X_train, X_train) + noise

    # L = cholesky(K)
    # t1 = 0.5 * y_train.T @ solve(L.T, solve(L, y_train))

    t1 = 0.5 * y_train.T @ solve(K, y_train)

    # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
    t2 = 0.5 * 2 * np.sum(np.log(np.diagonal(cholesky(K))))

    t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

    loglikelihood = t1 + t2 + t3

    # print(loglikelihood, kernel.l, kernel.sigma)

    # TODO: check this
    # assert loglikelihood >= 0, f"got negative log likelihood={loglikelihood}, t1={t1}, t2={t2}, t3={t3}"

    return loglikelihood


def compute_optimized_kernel(kernel, X_train, y_train):
    noise_level = 0.1

    def step(theta):
        return kernel_log_likelihood(kernel.set_params(theta), X_train, y_train, noise_level)

    default_params = kernel.default_params(X_train, y_train)

    if len(default_params) == 0:
        return kernel

    res = minimize(step,
                   default_params,
                   bounds=kernel.param_bounds(), method="L-BFGS-B", tol=0, options={"maxiter": 100})

    kernel.set_params(res.x)
    return kernel
