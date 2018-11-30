from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, cholesky, det, solve
from scipy.optimize import minimize

from myopt.kernels import Kernel


import torch


def mahalanobis_squared(xi, xj, VI=None):
    """Computes the pair-wise squared mahalanobis distance matrix as:
        (xi - xj)^T V^-1 (xi - xj)
    Args:
        xi (Tensor): xi input matrix.
        xj (Tensor): xj input matrix.
        VI (Tensor): The inverse of the covariance matrix, default: identity
            matrix.
    Returns:
        Weighted matrix of all pair-wise distances (Tensor).
    """
    if VI is None:
        xi_VI = xi
        xj_VI = xj
    else:
        xi_VI = xi @ VI
        xj_VI = xj @ VI

    D = (xi_VI * xi).sum(dim=-1).reshape(-1, 1) \
      + (xj_VI * xj).sum(dim=-1).reshape(1, -1) \
      - 2 * xi_VI @ xj

    return D.float()


class SQE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l = torch.nn.Parameter(torch.randn(1))
        self.sigma_s = torch.nn.Parameter(torch.randn(1))

    def forward(self, a, b):
        ls = (self.l**2).clamp(1e-5, 1e5)
        var_s = (self.sigma_s**2).clamp(1e-5, 1e5)

        M = (torch.eye(len(a)) * ls)
        dist = mahalanobis_squared(a, b, M)

        return var_s * (-0.5 * dist).exp()



def kernel_log_likelihood(kernel: Callable, noise_level: float, X_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
    noise = noise_level ** 2 * torch.eye(len(X_train))

    K = kernel(X_train, X_train) + noise

    Xsol, _ = torch.gesv(y_train, K)
    t1 = 0.5 * y_train @ Xsol

    s, tt2 = torch.slogdet(K)

    t2 = 0.5 * tt2
    t3 = 0.5 * len(X_train) * torch.log(2 * torch.tensor(np.pi))

    loglikelihood = t1 + t2 + t3

    # assert loglikelihood >= 0, f"got negative log likelihood={loglikelihood}, t1={t1}, t2={t2}, t3={t3}"
    # assert s == 1

    return loglikelihood


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

        # assert loglikelihood >= 0, f"got negative log likelihood={loglikelihood}, t1={t1}, t2={t2}, t3={t3}"
        # assert s == 1

        return loglikelihood

    return step


def plot_kernel_loss(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray, xmax=20) -> None:
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

    default_params = kernel.default_params(X_train, y_train)

    if len(default_params) == 0:
        return kernel

    res = minimize(step,
                   default_params,
                   bounds=kernel.param_bounds(), method="L-BFGS-B", tol=0, options={"maxiter": 100})

    # print(f"Kernel optimal params {res.x}")

    kernel.set_params(res.x)
    return kernel
    # return kernel.with_params(res.x)
