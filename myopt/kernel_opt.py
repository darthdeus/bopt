from typing import Callable

from scipy.optimize import minimize

import numpy as np
from numpy.linalg import inv, cholesky
import matplotlib.pyplot as plt

from .kernels import Kernel


def kernel_step(kernel: Kernel, noise_level: float, X_train: np.ndarray, y_train: np.ndarray) \
        -> Callable[[np.ndarray], np.ndarray]:

    def step(theta):
        noise = noise_level ** 2 * np.eye(len(X_train))
        K = kernel.with_params(theta)(X_train, X_train) + noise

        t1 = 0.5 * y_train.T @ inv(K) @ y_train
        t2 = 0.5 * np.linalg.det(K)
        t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

        return t1 + t2 + t3

    return step


def plot_kernel_loss(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray, xmax=5) -> None:
    noise_level = 0.1

    step = kernel_step(kernel, noise_level, X_train, y_train)

    X = np.arange(0.001, xmax, step=0.1)
    thetas = []

    for theta in X:
        # TODO: plot all params
        thetas.append(step(np.array([theta, 1])))

    thetas = np.array(thetas)

    plt.plot(X, thetas)


def compute_optimized_kernel(kernel, X_train, y_train):
    noise_level = 0.1

    step = kernel_step(kernel, noise_level, X_train, y_train)

    default_params = kernel.default_params()

    if len(default_params) == 0:
        return kernel

    res = minimize(step,
                   default_params,
                   bounds=kernel.param_bounds(), method="L-BFGS-B")

    print(f"Found optimal params {res.x}")

    return kernel.with_params(res.x)
