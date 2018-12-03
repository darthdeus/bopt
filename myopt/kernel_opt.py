from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, cholesky, det, solve
from scipy.optimize import minimize

from myopt.kernels import Kernel


def kernel_step(kernel: Kernel, noise_level: float, X_train: np.ndarray, y_train: np.ndarray) \
        -> Callable[[np.ndarray], np.ndarray]:
    def step(theta):
        noise = noise_level ** 2 * np.eye(len(X_train))
        kernel.set_params(theta)
        K = kernel(X_train, X_train) + noise

        # L = cholesky(K)

        t1 = 0.5 * y_train.T @ solve(K, y_train)
        # t1 = 0.5 * y_train.T @ solve(L.T, solve(L, y_train))
        # t1 = 0.5 * y_train.T @ inv(K) @ y_train

        # t2 = 0.5 * det(cholesky(K)) ** 2
        # t2 = 0.5 * det(K + 1e-6 * np.eye(len(K)))

        s, tt2 = np.linalg.slogdet(K)

        t2 = 0.5 * tt2
        t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

        loglikelihood = t1 + t2 + t3

        # print(loglikelihood, theta)

        # assert loglikelihood >= 0, f"got negative log likelihood={loglikelihood}, t1={t1}, t2={t2}, t3={t3}"
        # assert s == 1

        return loglikelihood

    return step


def plot_kernel_loss(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray, xmax=20) -> None:
    noise_level = 0.1

    step = kernel_step(kernel, noise_level, X_train, y_train)

    X = np.arange(0.00001, xmax, step=0.1)
    thetas = []

    for theta in X:
        # TODO: plot all params
        thetas.append(step(np.array([theta, kernel.sigma])))

    thetas = np.array(thetas)

    plt.plot(X, thetas)


def compute_optimized_kernel(kernel, X_train, y_train):
    noise_level = 0.1

    step = kernel_step(kernel, noise_level, X_train, y_train)

    default_params = kernel.default_params(X_train, y_train)

    if len(default_params) == 0:
        return kernel

    res = minimize(step,
                   default_params,
                   bounds=kernel.param_bounds(), method="L-BFGS-B", tol=0, options={"maxiter": 100})

    # print(f"Kernel optimal params {res.x}")

    kernel.set_params(res.x)
    return kernel
