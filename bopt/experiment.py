import yaml
import os
import psutil
import time
import pathlib
import numpy as np

from typing import List

from bopt.models.model import Model
from bopt.models.random_search import RandomSearch
from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job, Runner
from bopt.models.model import Sample, SampleCollection
from bopt.models.model import Model

from bopt.optimization_result import OptimizationResult


# TODO: fix numpy being stored everywhere when serializing! possibly in hyp.sample? :(
class Experiment:
    hyperparameters: List[Hyperparameter]
    runner: Runner
    samples: List[Sample]

    def __init__(self, hyperparameters: List[Hyperparameter], runner: Runner):
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.samples = []

    def run_next(self, model: Model, meta_dir: str, output_dir: str) -> Job:
        if len(self.samples) == 0:
            model = RandomSearch()

        sample_collection = SampleCollection(self.ok_samples(), meta_dir)

        # TODO: pridat normalizaci
        next_params, fitted_model = \
                model.predict_next(self.hyperparameters, sample_collection)

        job = self.runner.start(output_dir, next_params)

        next_sample = Sample(next_params, job, fitted_model)

        self.samples.append(next_sample)

        return job

    def run_loop(self, model: Model, meta_dir: str, n_iter=20) -> None:
        # TODO: ...
        print("running")

        output_dir = pathlib.Path(meta_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_iter):
            job = self.run_next(model, meta_dir, str(output_dir))

            while not job.is_finished():
                psutil.wait_procs(psutil.Process().children(), timeout=0.01)
                time.sleep(1)

            self.serialize(meta_dir)

    def to_serializable(self) -> "Experiment":
        samples = [s.to_serializable() for s in self.samples]
        exp = Experiment(self.hyperparameters, self.runner)
        exp.samples = samples
        return exp

    def from_serializable(self) -> "Experiment":
        samples = [s.from_serializable() for s in self.samples]
        exp = Experiment(self.hyperparameters, self.runner)
        exp.samples = samples
        return exp

    def serialize(self, meta_dir: str) -> None:
        dump = yaml.dump(self.to_serializable())

        with open(os.path.join(meta_dir, "meta.yml"), "w") as f:
            f.write(dump)

    @staticmethod
    def deserialize(meta_dir: str) -> "Experiment":
        with open(os.path.join(meta_dir, "meta.yml"), "r") as f:
            contents = f.read()
            obj = yaml.load(contents)

        if obj.samples is None:
            obj.samples = []

        return obj.from_serializable()

    def ok_samples(self) -> List[Sample]:
        return [s for s in self.samples if s.job.is_finished()]

    def current_optim_result(self, meta_dir: str) -> OptimizationResult:
        sample_col = SampleCollection(self.ok_samples(), meta_dir)

        X_sample, y_sample = sample_col.to_xy()

        # TODO: this should be handled better
        params = sorted(self.hyperparameters, key=lambda h: h.name)

        # TODO; normalizace?
        y_sample = (y_sample - y_sample.mean()) / y_sample.std()

        best_y = None
        best_x = None

        if len(y_sample) > 0:
            best_idx = np.argmax(y_sample)
            best_y = y_sample[best_idx]
            best_x = X_sample[best_idx]

        # TODO: fuj
        from bopt.kernels.kernels import SquaredExp

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

