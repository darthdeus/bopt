import abc
import datetime

import numpy as np
from numpy.linalg import inv, cholesky, solve
from typing import Optional, Tuple, List

import bopt.kernels.kernel_opt as kernel_opt
from bopt.kernels.kernels import Kernel, SquaredExp
from bopt.plot import plot_gp
from bopt.models.model import Model


def convert_to_rank1(arr):
    assert arr.ndim == 2, f"rank-2 tensor required, got {arr.ndim}"
    assert arr.shape[1] == 1, f"arr.shape[1] must equal `1`, got {arr.shape}"

    return arr.squeeze()


class GaussianProcessRegressor:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: Optional[np.ndarray]

    mu: np.ndarray
    cov: np.ndarray
    std: np.ndarray
    noise: float

    kernel: Kernel
    stable_computation: bool

    def __init__(self, noise=0, kernel=SquaredExp(), stable_computation=True):
        self.noise = noise
        self.kernel = kernel

        self.X_train = None
        self.y_train = None
        self.X_test = None

        self.mu = None
        self.cov = None
        self.std = None

        self.stable_computation = stable_computation

    def to_serializable(self) -> "GaussianProcessRegressor":
        model = self.copy()

        model.noise = float(model.noise)

        model.mu = None
        model.cov = None
        model.std = None
        model.X_train = None
        model.y_train = None
        model.X_test = None

        model.kernel = model.kernel.to_serializable()
        return model

    def from_serializable(self) -> "GaussianProcessRegressor":
        model = self.copy()
        model.kernel = model.kernel.from_serializable()
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, kernel=None) -> "GaussianProcess":
        assert X_train.ndim == 2, f"X_train must be rank-2 tensor, got ndim={X_train.ndim}"

        if kernel is not None:
            self.kernel = kernel

        self.X_train = X_train
        self.y_train = y_train

        return self

    def posterior(self, X_test: np.ndarray, X_train: np.ndarray = None, y_train: np.ndarray = None) -> "GaussianProcess":
        """
        Computes the posterior `p(y_test | X_test, X_train, y_train)`
        and stores the result.
        """
        if (X_train is not None) and (y_train is not None):
            self.fit(X_train, y_train)

        assert X_test.ndim == 2, f"X_test must be rank-2 tensor, got ndim={X_test.ndim}"

        assert self.X_train is not None, "X_train is None, call `fit` first, or provide X_train directly"
        assert self.y_train is not None, "y_train is None, call `fit` first, or provide y_train directly"

        self.X_test = X_test

        assert self.y_train.ndim == 1
        assert self.y_train.shape[0] == self.X_train.shape[0]
        assert self.X_test is not None

        if self.X_train.ndim > 1:
            assert self.X_train.shape[1] == self.X_test.shape[1], \
                    f"got {self.X_train.shape} and {self.X_test.shape}"

        noise = (self.noise ** 2) * np.eye(len(self.X_train))

        # TODO: get rid of numpy
        # print("XXX noise", noise)
        K = self.kernel(self.X_train, self.X_train).numpy() + noise
        K_s = self.kernel(self.X_train, X_test).numpy()
        K_ss = self.kernel(X_test, X_test).numpy()

        K_ss_noise = (self.noise ** 2) * np.eye(len(K_ss))
        # Symmetrization hack
        # K = (K + K.T) / 2.0
        # K_ss = (K_ss + K_ss.T) / 2.0

        eye_eps = 1e-8

        K_stable_eye   = eye_eps * np.eye(len(K))  # Just for numerical stability?
        # Ks_stable_eye  = eye_eps * np.eye(len(K_s))  # Just for numerical stability?
        Kss_stable_eye = eye_eps * np.eye(len(K_ss))  # Just for numerical stability?

        for i in range(min(*K_s.shape)):
            K_s[i, i] += eye_eps

        K += K_stable_eye
        # K_s += Ks_stable_eye
        K_ss += Kss_stable_eye

        assert np.allclose(K, K.T, atol=1e-7), "K is not symmetric"
        assert np.allclose(K_ss, K_ss.T, atol=1e-7), "K_ss is not symmetric"

        # eigs = np.linalg.eigvals(K)
        # if np.any(eigs <= 0):
        #     print("Got negative eigs", eigs)

        if self.stable_computation:
            # print("y_train", self.X_train)
            # print()

            L = cholesky(K)
            alpha = solve(L.T, solve(L, self.y_train))
            assert np.allclose((L @ L.T) @ alpha, self.y_train)

            alpha = solve(K, self.y_train)
            assert np.allclose(K @ alpha, self.y_train)
            L_k = solve(L, K_s)

            # print("HA")
            # print(K_s[:5, :5])
            # print(alpha)
            # print(alpha.shape, K_s.shape)

            self.mu = K_s.T @ alpha
            self.cov = K_ss - L_k.T @ L_k + K_ss_noise
            self.std = np.sqrt(np.diag(K_ss) - np.sum(L_k ** 2, axis=0))
        else:
            K_inv = inv(K)
            self.mu = K_s.T @ K_inv @ self.y_train
            self.cov = K_ss - K_s.T @ K_inv @ K_s
            self.std = np.sqrt(np.diag(self.cov))

        assert self.mu.ndim == 1
        assert self.mu.shape[0] == X_test.shape[0]
        assert self.cov.shape == (X_test.shape[0], X_test.shape[0])

        return self

    def optimize_kernel(self) -> "GaussianProcess":
        assert self.X_train is not None, "X_train is None, call `fit` first"
        assert self.y_train is not None, "y_train is None, call `fit` first"

        self.kernel, self.noise = kernel_opt.compute_optimized_kernel(
                                      self.kernel,
                                      self.X_train,
                                      self.y_train)

        return self

    def log_prob(self) -> float:
        nll = kernel_opt.kernel_log_likelihood(self.kernel, self.X_train, self.y_train,
                    noise_level=self.noise)

        return nll

    def plot_prior(self, X, **kwargs):
        plot_gp(np.zeros(len(X)), self.kernel(X, X), X, kernel=self.kernel, **kwargs)

        return self

    def plot_posterior(self, **kwargs):
        assert self.X_test is not None, "X_test was not provided, call `.posterior(X_test)` first"

        plot_gp(self.mu, self.cov,
                convert_to_rank1(self.X_test),
                convert_to_rank1(self.X_train),
                self.y_train,
                kernel=self.kernel, noise=self.noise, nll=round(self.log_prob().numpy().item(), 3), **kwargs)

        return self

    def copy(self) -> "GaussianProcessRegressor":
        gp = GaussianProcessRegressor()
        gp.kernel = self.kernel.copy()
        gp.noise = self.noise

        gp.X_train = self.X_train
        gp.y_train = self.y_train
        gp.X_test = self.X_test

        gp.mu = self.mu
        gp.cov = self.cov
        gp.std = self.std

        return gp

    def with_kernel(self, kernel: Kernel) -> "GaussianProcess":
        gp = self.copy()
        gp.kernel = kernel

        return gp

    def with_kernel_params(self, theta) -> "GaussianProcess":
        gp = self.copy()
        gp.kernel = gp.kernel.copy()
        gp.kernel.set_params(theta)

        return gp

    def with_noise(self, noise: float) -> "GaussianProcess":
        gp = self.copy()
        gp.noise = noise

        return gp

    def mu_std(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mu, self.std
