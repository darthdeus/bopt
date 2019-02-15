import os
import pathlib
import yaml
import re
import numpy as np
import pickle

from glob import glob
from typing import Union, NamedTuple, List, Any, Optional, Tuple

from bopt.gaussian_process import GaussianProcess
from bopt.basic_types import Hyperparameter
from bopt.kernels import SquaredExp, Kernel
from bopt.runner.abstract import Job, Runner
from bopt.sample import Sample


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


# TODO: fix numpy being stored everywhere when serializing! possibly in hyp.sample? :(
class Experiment:
    hyperparameters: List[Hyperparameter]
    runner: Runner
    samples: List[Sample]

    def __init__(self, hyperparameters: List[Hyperparameter], runner: Runner):
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.samples = []

        # pathlib.Path(os.path.join(self.meta_dir, "outputs"))\
        #         .mkdir(parents=True, exist_ok=True)
        # self.serialize()

    def serialize(self, meta_dir) -> None:
        dump = yaml.dump(self)

        with open(os.path.join(meta_dir, "meta.yml"), "w") as f:
            f.write(dump)

    @staticmethod
    def deserialize(meta_dir: str) -> "Experiment":
        with open(os.path.join(meta_dir, "meta.yml"), "r") as f:
            contents = f.read()
            obj = yaml.load(contents)

        return obj

    def current_optim_result(self) -> OptimizationResult:
        finished_evaluations = [e for e in self.samples if e.is_success()]

        # TODO: this should be handled better
        params = sorted(self.hyperparameters, key=lambda h: h.name)

        X_sample_vals = [e.sorted_parameter_values() for e in finished_evaluations]

        X_sample = np.array(X_sample_vals)
        y_sample = np.array([e.final_result() for e in finished_evaluations])

        # TODO; normalizace?
        y_sample = (y_sample - y_sample.mean()) / y_sample.std()

        best_y = None
        best_x = None
        if len(y_sample) > 0:
            best_idx = np.argmax(y_sample)
            best_y = y_sample[best_idx]
            best_x = X_sample[best_idx]

        kernel = SquaredExp()
        n_iter = len(X_sample)

        return OptimizationResult(
                X_sample,
                y_sample,
                best_x,
                best_y,
                params,
                kernel,
                n_iter,
                opt_fun=None)

