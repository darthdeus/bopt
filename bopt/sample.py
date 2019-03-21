import os
import logging
import traceback
import numpy as np

from typing import Tuple, List, Optional

from bopt.basic_types import Hyperparameter, JobStatus
from bopt.job_params import JobParams
from bopt.runner.abstract import Job
from bopt.runner.job_loader import JobLoader
from bopt.models.parameters import ModelParameters


class Sample:
    job: Job
    result: Optional[float]
    model: ModelParameters
    mu_pred: float
    sigma_pred: float

    def __init__(self, job: Job, model_params: ModelParameters,
            mu_pred: float, sigma_pred: float) -> None:
        self.job = job
        self.model = model_params
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        self.result = None

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict(),
            "model": self.model.to_dict() if self.model is not None else None,
            "result": self.result,
            "mu_pred": self.mu_pred,
            "sigma_pred": self.sigma_pred
        }

    def status(self) -> JobStatus:
        if self.result is not None:
            return JobStatus.FINISHED
        elif self.job is not None:
            return self.job.status()
        else:
            logging.error("Somehow created a sample with no job and no result.")
            return JobStatus.FAILED

    @staticmethod
    def from_dict(data: dict, hyperparameters: List[Hyperparameter]) -> "Sample":
        model_dict = None

        if data["model"] is not None:
            model_dict = ModelParameters.from_dict(data["model"])

        job = JobLoader.from_dict(data["job"], hyperparameters)
        sample = Sample(job,
                model_dict,
                data["mu_pred"],
                data["sigma_pred"])
        sample.result = data["result"]

        return sample

    def to_x(self) -> np.ndarray:
        return self.job.run_parameters.x
        # return Sample.param_dict_to_x(self.job.run_parameters)

    def to_xy(self, output_dir: str) -> Tuple[np.ndarray, float]:
        x = self.to_x()

        status = self.status()

        assert self.result or self.job

        if status == JobStatus.FINISHED:
            # TODO: collect first?
            y = self.result or self.job.get_result(output_dir)
        elif status == JobStatus.RUNNING:
            logging.info("Using mean prediction for a running job {}".format(self.job.job_id))
            y = self.mu_pred
        else:
            logging.error("Tried to get xy for a job {} which is {}".format(self.job.job_id, status))
            raise ValueError(self)

        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

        return x, y

    def get_result(self, meta_dir: str) -> float:
        # TODO: fuj
        output_dir = os.path.join(meta_dir, "output")
        return self.job.get_result(output_dir)

    def __str__(self) -> str:
        s = f"{self.job.job_id}\t"
        is_finished = self.job.is_finished()

        if self.job.is_finished():
            # TODO: handle failed
            # if self.is_success():
                # TODO: meta_dir not needed since we're always cding?
                final_result = self.get_result(".")

                rounded_params = {h.name: value for h, value in
                        self.job.run_parameters.mapping.items()}

                assert isinstance(final_result, float)
                s += f"{is_finished}\t{final_result:.3f}\t{rounded_params}"
            # else:
            #     s += f"FAILED: {self.err()}"
        else:
            s += "RUNNING"

        return s



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
