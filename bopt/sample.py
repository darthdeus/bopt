import os
import numpy as np

from typing import Tuple, List

from bopt.runner.abstract import Job
from bopt.runner.job_loader import JobLoader
from bopt.models.parameters import ModelParameters


class Sample:
    job: Job
    model: ModelParameters

    def __init__(self, job: Job, fitted_model: ModelParameters) -> None:
        self.job = job
        self.model = fitted_model

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict(),
            "model": self.model.to_dict() if self.model is not None else None,
        }

    @staticmethod
    def from_dict(data: dict) -> "Sample":
        model_dict = None

        if data["model"] is not None:
            model_dict = ModelParameters.from_dict(data["model"])

        return Sample(JobLoader.from_dict(data["job"]),
                      model_dict)

    def to_x(self) -> np.ndarray:
        param_values = self.job.run_parameters

        x = np.zeros(len(param_values), dtype=np.float64)

        for i, key in enumerate(sorted(param_values)):
            value = param_values[key]

            x[i] = value

        return x

    def to_xy(self, output_dir: str) -> Tuple[np.ndarray, float]:
        x = self.to_x()

        y = self.job.get_result(output_dir)

        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

        return x, y

    def get_result(self, meta_dir: str) -> float:
        import os
        # TODO: fuj
        output_dir = os.path.join(meta_dir, "output")
        return self.job.get_result(output_dir)


class SampleCollection:
    samples: List[Sample]
    meta_dir: str

    def __init__(self, samples: List[Sample], meta_dir: str) -> None:
        self.samples = samples
        self.meta_dir = meta_dir

    def to_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        assert all([s.job.is_finished() for s in self.samples])

        num_samples = len(self.samples)

        y_sample = np.zeros([num_samples], dtype=np.float64)
        # TODO: chci to normalizovat?
        zero_mean = y_sample - y_sample.mean()

        std = y_sample.std()

        if std > 0:
            y_sample = zero_mean / std
        else:
            y_sample = zero_mean

        xs = []

        # TODO: "output" as a global constant
        output_dir = os.path.join(self.meta_dir, "output")

        for i, sample in enumerate(self.samples):
            x, y = sample.to_xy(output_dir)

            xs.append(x)
            y_sample[i] = y

        X_sample = np.array(xs, dtype=np.float32)

        return X_sample, y_sample
