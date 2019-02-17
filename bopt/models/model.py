import abc
import numpy as np

from typing import List, Tuple
from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job


class Model(abc.ABC):
    @abc.abstractmethod
    def predict_next(self, samples: "SampleCollection") -> Tuple[dict, "Model"]:
        pass


class Sample:
    model: Model
    param_values: dict
    job: Job

    def __init__(self, param_values: dict, job: Job, fitted_model: Model) -> None:
        self.param_values = param_values
        self.job = job
        self.model = fitted_model

    def to_xy(self, output_dir: str) -> Tuple[np.ndarray, float]:
        x = np.zeros(len(self.param_values), dtype=np.float64)

        for i, key in enumerate(sorted(self.param_values)):
            value = self.param_values[key]

            x[i] = value

        y = self.job.get_result(output_dir)

        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

        return x, y


class SampleCollection:
    samples: List[Sample]
    meta_dir: str
    output_dir: str

    def __init__(self, samples: List[Sample], meta_dir: str, output_dir: str) -> None:
        self.samples = samples
        self.meta_dir = meta_dir
        self.output_dir = output_dir
