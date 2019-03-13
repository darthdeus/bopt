import os
import logging
import numpy as np

from typing import Tuple, List

from bopt.job_params import JobParams
from bopt.runner.abstract import Job
from bopt.runner.job_loader import JobLoader
from bopt.models.parameters import ModelParameters


class Sample:
    job: Job
    model: ModelParameters
    mu_pred: float
    sigma_pred: float

    def __init__(self, job: Job, model_params: ModelParameters,
            mu_pred: float, sigma_pred: float) -> None:
        self.job = job
        self.model = model_params
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict(),
            "model": self.model.to_dict() if self.model is not None else None,
            "mu_pred": self.mu_pred,
            "sigma_pred": self.sigma_pred
        }

    @staticmethod
    def from_dict(data: dict) -> "Sample":
        model_dict = None

        if data["model"] is not None:
            model_dict = ModelParameters.from_dict(data["model"])

        return Sample(JobLoader.from_dict(data["job"]),
                      model_dict,
                      data["mu_pred"],
                      data["sigma_pred"])

    def to_x(self) -> np.ndarray:
        return self.job.run_parameters.x
        # return Sample.param_dict_to_x(self.job.run_parameters)

    def to_xy(self, output_dir: str) -> Tuple[np.ndarray, float]:
        x = self.to_x()

        if self.job.is_finished():
            y = self.job.get_result(output_dir)
        else:
            logging.warning(f"Using mean prediction for sample {self}")
            y = self.mu_pred

        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

        return x, y

    def get_result(self, meta_dir: str) -> float:
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
        num_samples = len(self.samples)

        y_sample = np.zeros([num_samples], dtype=np.float64)
        # TODO: chci to normalizovat?
        # zero_mean = y_sample - y_sample.mean()
        #
        # std = y_sample.std()
        #
        # if std > 0:
        #     y_sample = zero_mean / std
        # else:
        #     y_sample = zero_mean

        xs = []

        # TODO: "output" as a global constant
        output_dir = os.path.join(self.meta_dir, "output")

        for i, sample in enumerate(self.samples):
            x, y = sample.to_xy(output_dir)

            xs.append(x)
            y_sample[i] = y

        X_sample = np.array(xs, dtype=np.float32)
        Y_sample = y_sample.reshape(-1, 1)

        return X_sample, Y_sample
