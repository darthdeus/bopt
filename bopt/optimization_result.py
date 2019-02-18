import os
import pathlib
import yaml
import re
import numpy as np
import pickle

from glob import glob
from typing import Union, NamedTuple, List, Any, Optional, Tuple

from bopt.basic_types import Hyperparameter
from bopt.kernels.kernels import Kernel
from bopt.runner.abstract import Job, Runner
from bopt.models.model import Sample
from bopt.models.gaussian_process_regressor import GaussianProcessRegressor


class OptimizationResult:
    X_sample: np.ndarray
    y_sample: np.ndarray
    best_x: Optional[np.ndarray]
    best_y: Optional[float]
    params: List[Hyperparameter]
    kernel: Kernel
    n_iter: int
    opt_fun: Any

    def __init__(self, X_sample: np.ndarray, y_sample: np.ndarray,
            best_x: Optional[np.ndarray], best_y: Optional[float],
            params: List[Hyperparameter], kernel: Kernel, n_iter: int, opt_fun: Any) -> None:
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.best_x = best_x
        self.best_y = best_y
        self.params = params
        self.kernel = kernel
        self.n_iter = n_iter
        self.opt_fun = opt_fun

    def __repr__(self) -> str:
        # TODO: name bounds
        # [f"{name}={round(val, 3)}" for name, val in zip(self.bounds, self.best_x)]
        return f"OptimizationResult(best_x={self.best_x}, best_y={self.best_y})"

    def dump(self, filename) -> None:
        with open(filename, "wb") as f:
            opt_fun = self.opt_fun
            self.opt_fun = None
            pickle.dump(self, filename)
            self.opt_fun = opt_fun

    def slice_at(self, i: int, gp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bound = self.params[i].range
        resolution = 15
        x_i = np.linspace(bound.low, bound.high, resolution)

        X_test = np.tile(self.best_x, (resolution, 1))
        X_test[:, i] = x_i

        mu, std = gp.posterior(X_test).mu_std()

        return x_i, mu, std

    @staticmethod
    def load(filename) -> "OptimizationResult":
        with open(filename, "rb") as f:
            return pickle.load(f)


    def fit_gp(self) -> "GaussianProcessRegressor":
        gp = GaussianProcessRegressor(kernel=self.kernel) \
                .fit(self.X_sample, self.y_sample) \
                .optimize_kernel()

        return gp
