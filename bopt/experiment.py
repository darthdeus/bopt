import yaml
import os
import numpy as np
import pathlib
from typing import List

from bopt.models.model import Model
from bopt.models.random_search import RandomSearch
from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job, Runner
from bopt.models.model import Sample, SampleCollection
from bopt.models.model import Model


# TODO: fix numpy being stored everywhere when serializing! possibly in hyp.sample? :(
class Experiment:
    hyperparameters: List[Hyperparameter]
    runner: Runner
    samples: List[Sample]

    def __init__(self, hyperparameters: List[Hyperparameter], runner: Runner):
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.samples = []

    def run_next(self, model: Model, meta_dir: str, output_dir: str) -> None:
        if len(self.samples) == 0:
            model = RandomSearch()

        sample_collection = SampleCollection(self.samples, meta_dir, output_dir)

        next_params, fitted_model = \
                model.predict_next(self.hyperparameters, sample_collection)

        job = self.runner.start(output_dir, next_params)

        next_sample = Sample(next_params, job, fitted_model)

        self.samples.append(next_sample)

    def run_loop(self, model: Model, meta_dir: str, n_iter=20) -> None:
        # TODO: ...
        print("running")

        output_dir = pathlib.Path(meta_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.run_next(model, meta_dir, str(output_dir))

        self.serialize(meta_dir)

    def serialize(self, meta_dir) -> None:
        dump = yaml.dump(self)

        with open(os.path.join(meta_dir, "meta.yml"), "w") as f:
            f.write(dump)

    @staticmethod
    def deserialize(meta_dir: str) -> "Experiment":
        with open(os.path.join(meta_dir, "meta.yml"), "r") as f:
            contents = f.read()
            obj = yaml.load(contents)

        if obj.samples is None:
            obj.samples = []

        return obj

    def ok_samples(self) -> List[Sample]:
        pass

    # def current_optim_result(self) -> OptimizationResult:
    #     samples = [e for e in self.ok_samples()]
    #
    #     # TODO: this should be handled better
    #     params = sorted(self.hyperparameters, key=lambda h: h.name)
    #
    #     X_sample_vals = [e.sorted_parameter_values() for e in samples]
    #
    #     X_sample = np.array(X_sample_vals)
    #     y_sample = np.array([e.final_result() for e in samples])
    #
    #     # TODO; normalizace?
    #     y_sample = (y_sample - y_sample.mean()) / y_sample.std()
    #
    #     best_y = None
    #     best_x = None
    #
    #     if len(y_sample) > 0:
    #         best_idx = np.argmax(y_sample)
    #         best_y = y_sample[best_idx]
    #         best_x = X_sample[best_idx]
    #
    #     kernel = SquaredExp()
    #     n_iter = len(X_sample)
    #
    #     return OptimizationResult(
    #             X_sample,
    #             y_sample,
    #             best_x,
    #             best_y,
    #             params,
    #             kernel,
    #             n_iter,
    #             opt_fun=None)
    #
