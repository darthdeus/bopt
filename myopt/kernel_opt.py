from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, cholesky, det, solve
from scipy.optimize import minimize

from functools import partial

from myopt.kernels import Kernel
from myopt.plot import imshow


def kernel_log_likelihood(kernel: Kernel, X_train: np.ndarray,
                          y_train: np.ndarray, noise_level: float = 0) -> float:
    noise = noise_level ** 2 * np.eye(len(X_train))
    K = kernel(X_train, X_train) + noise

    # L = cholesky(K)

    t1 = 0.5 * y_train.T @ solve(K, y_train)
    # t1 = 0.5 * y_train.T @ solve(L.T, solve(L, y_train))

    # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
    t2 = 0.5 * 2 * np.sum(np.log(np.diagonal(cholesky(K))))

    t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

    loglikelihood = t1 + t2 + t3

    # print(loglikelihood, kernel.l, kernel.sigma)

    # TODO: check this
    # assert loglikelihood >= 0, f"got negative log likelihood={loglikelihood}, t1={t1}, t2={t2}, t3={t3}"

    return loglikelihood


def plot_kernel_loss(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray,
                     noise_level: float = 0.1, xmax: int = 5, sigma: float = 1) -> None:
    X = np.linspace(0.00001, xmax, num=50)

    def likelihood(l):
        return kernel_log_likelihood(kernel.set_params(np.array([l, sigma])),
                                                         X_train, y_train, noise_level)

    data = np.vectorize(likelihood)(X)

    plt.plot(X, data)
    plt.title(f"Kernel marginal likelihood, $\sigma = {sigma}$")


def plot_kernel_loss_2d(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray,
                        noise_level: float = 0.1) -> None:
    num_points = 10
    data = np.zeros((num_points, num_points))

    amin = 0.3
    amax = 5

    bmin = 1
    bmax = 10

    a_values = np.linspace(amin, amax, num=num_points)
    b_values = np.linspace(bmin, bmax, num=num_points)

    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            theta = np.array([a, b])
            data[i, j] = kernel_log_likelihood(kernel.set_params(theta), X_train, y_train, noise_level)

    imshow(data, a_values, b_values)


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
