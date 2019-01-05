import os
import pathlib
import yaml
import re
import numpy as np
import pickle

from glob import glob
from typing import Union, NamedTuple, List, Any

from myopt.kernels import SquaredExp, Kernel
from myopt.runner.abstract import Job, Runner

class Integer:
    low: int
    high: int

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.type = "int"

    def sample(self) -> float:
        return np.random.randint(self.low, self.high)

    def __repr__(self) -> str:
        return f"Int({self.low}, {self.high})"

class Float:
    low: float
    high: float

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.type = "float"

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Float({self.low}, {self.high})"

# TODO: hehe
Range = Union[Integer, Float]
Bound = Union[Integer, Float]


class Hyperparameter(NamedTuple):
  name: str
  range: Range


class OptimizationResult:
    X_sample: np.ndarray
    y_sample: np.ndarray
    best_x: np.ndarray
    best_y: float
    bounds: List[Bound]
    kernel: Kernel
    n_iter: int
    opt_fun: Any

    def __init__(self, X_sample: np.ndarray, y_sample: np.ndarray, best_x: np.ndarray, best_y: float,
            bounds: List[Bound], kernel: Kernel, n_iter: int, opt_fun: Any) -> None:
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.best_x = best_x
        self.best_y = best_y
        self.bounds = bounds
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
        X_sample = np.array([list(e.run_parameters.values()) for e in self.evaluations])
        y_sample = np.array([e.final_result() for e in self.evaluations])

        best_y = np.max(y_sample)
        best_x = X_sample[np.argmax(best_y)]

        kernel = SquaredExp()
        n_iter = len(X_sample)

        bounds = [h.range for h in self.hyperparameters]

        return OptimizationResult(
                X_sample,
                y_sample,
                best_x,
                best_y,
                bounds,
                kernel,
                n_iter,
                opt_fun=None)

