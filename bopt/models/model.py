import abc

from typing import List, Tuple
from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job


class Model(abc.ABC):
    def predict_next(self, samples: List["Sample"]) -> Tuple[dict, "Model"]:
        pass


class Sample:
    model: Model
    param_values: dict
    job: Job

    def __init__(self, param_values: dict, job: Job, fitted_model: Model) -> None:
        self.param_values = param_values
        self.job = job
        self.model = fitted_model
