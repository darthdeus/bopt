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

    def slice_at(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        gp = GaussianProcess(kernel=self.kernel.copy())
        gp.fit(self.X_sample, self.y_sample)

        bound = self.params[i].range
        resolution = 30
        x_i = np.linspace(bound.low, bound.high, resolution)

        X_test = np.tile(self.best_x, (resolution, 1))
        X_test[:, i] = x_i

        mu, _ = gp.posterior(X_test).mu_std()

        return x_i, mu

    @staticmethod
    def load(filename) -> "OptimizationResult":
        with open(filename, "rb") as f:
            return pickle.load(f)

META_FILENAME = "meta.yml"


class Experiment:
    meta_dir: str
    hyperparameters: List[Hyperparameter]
    runner: Runner
    evaluations: List[Job]

    def __init__(self, meta_dir: str, hyperparameters: List[Hyperparameter], runner: Runner):
        self.meta_dir = meta_dir
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.evaluations = []

        pathlib.Path(os.path.join(self.meta_dir, "outputs"))\
                .mkdir(parents=True, exist_ok=True)

        self.serialize()

    def iterate(self):
        eval_params = [param.sample() for param in self.hyperparameters]

        job = self.runner.start(eval_params)

        self.evaluations.append(job)

    @staticmethod
    def filename(directory) -> str:
        return os.path.join(directory, META_FILENAME)

    def serialize(self) -> None:
        evals = self.evaluations
        del self.evaluations
        dump = yaml.dump(self)
        self.evaluations = evals

        with open(Experiment.filename(self.meta_dir), "w") as f:
            f.write(dump)

        for job in self.evaluations:
            job.serialize()

    @staticmethod
    def deserialize(directory: str) -> "Experiment":
        with open(Experiment.filename(directory), "r") as f:
            contents = f.read()
            obj = yaml.load(contents)

        jobs = []
        for path in glob(os.path.join(directory, "job-*")):
            matches = re.match('.*?(\d+).*?', path)
            assert matches is not None

            job_id = int(matches.group(1))
            job = obj.runner.deserialize_job(obj.meta_dir, job_id)
            job.deserialize()

            jobs.append(job)
        obj.evaluations = jobs

        return obj

    def current_optim_result(self) -> OptimizationResult:
        finished_evaluations = [e for e in self.evaluations if e.is_success()]

        # TODO: this should be handled better
        params = sorted(self.hyperparameters, key=lambda h: h.name)

        X_sample = np.array([e.sorted_parameter_values() for e in finished_evaluations])
        y_sample = np.array([e.final_result() for e in finished_evaluations])

        best_y = None
        best_x = None
        if len(y_sample) > 0:
            best_y = np.max(y_sample)
            best_x = X_sample[np.argmax(best_y)]

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

