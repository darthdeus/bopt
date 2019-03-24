import os
import logging
import traceback
import numpy as np

from typing import Tuple, List, Optional

from bopt.basic_types import Hyperparameter, JobStatus
from bopt.hyperparam_values import HyperparamValues
from bopt.runner.abstract import Job
from bopt.runner.job_loader import JobLoader
from bopt.models.parameters import ModelParameters


class Sample:
    job: Optional[Job]
    model: ModelParameters
    hyperparam_values: HyperparamValues
    result: Optional[float]
    mu_pred: float
    sigma_pred: float
    comment: Optional[str]
    waiting_for_similar: bool

    def __init__(self, job: Optional[Job],
            model_params: ModelParameters,
            hyperparam_values: HyperparamValues,
            mu_pred: float,
            sigma_pred: float) -> None:
        self.job = job
        self.model = model_params
        self.hyperparam_values = hyperparam_values
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        self.result = None
        self.comment = None
        self.waiting_for_similar = False

    def status(self) -> JobStatus:
        if self.waiting_for_similar:
            return JobStatus.WAITING_FOR_SIMILAR
        elif self.result is not None:
            return JobStatus.FINISHED
        elif self.job is not None:
            if self.job.is_finished():
                return JobStatus.FAILED
            else:
                return JobStatus.RUNNING
        else:
            logging.error("Somehow created a sample with no job and no result.")
            return JobStatus.FAILED

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict() if self.job else None,
            "model": self.model.to_dict() if self.model else None,
            "hyperparam_values": self.hyperparam_values.to_dict(),
            "result": self.result,
            "mu_pred": self.mu_pred,
            "sigma_pred": self.sigma_pred,
            "comment": self.comment,
            "waiting_for_similar": self.waiting_for_similar
        }

    @staticmethod
    def from_dict(data: dict, hyperparameters: List[Hyperparameter]) -> "Sample":
        model_dict = None

        if data["model"]:
            model_dict = ModelParameters.from_dict(data["model"])


        # TODO: 64 or 32 bit?
        x = np.array(data["hyperparam_values"], dtype=np.float64)
        hyperparam_values = HyperparamValues.mapping_from_vector(x, hyperparameters)

        if "job" in data and data["job"] is not None:
            job = JobLoader.from_dict(data["job"])
        else:
            job = None

        sample = Sample(job,
                model_dict,
                hyperparam_values,
                data["mu_pred"],
                data["sigma_pred"])

        sample.waiting_for_similar = data["waiting_for_similar"]
        sample.comment = data.get("comment", None)
        sample.result = data.get("result", None)

        return sample

    def to_x(self) -> np.ndarray:
        return self.hyperparam_values.x

    def to_xy(self) -> Tuple[np.ndarray, float]:
        x = self.to_x()

        status = self.status()

        assert self.result or self.job

        if status == JobStatus.FINISHED:
            # TODO: collect first?
            # TODO: TADY SEM SE VRATIT :PPP
            y = self.result
        elif status == JobStatus.WAITING_FOR_SIMILAR:
            y = self.mu_pred
        elif status == JobStatus.RUNNING:
            if self.job:
                logging.info("Using mean prediction for a running job {}".format(self.job.job_id))
                y = self.mu_pred
            else:
                raise ValueError("Sample has status RUNNING but no job.")
        else:
            logging.error("Tried to get xy for a sample {} which is {}".format(self, status))
            raise ValueError(self)

        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

        return x, y

    def __str__(self) -> str:
        if self.job:
            s = f"{self.job.job_id}\t"
        else:
            s = "manual\t"

        is_finished = self.status() == JobStatus.FINISHED

        # TODO: proper status check

        if self.result:
            rounded_params = {h.name: (round(v, 3) if isinstance(v, float) else v)
                    for h, v in self.hyperparam_values.mapping.items()}

            assert isinstance(self.result, float)
            s += f"{is_finished}\t{self.result:.3f}\t{rounded_params}"
        else:
            s += str(self.status())

        return s


class SampleCollection:
    samples: List[Sample]

    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

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

        for i, sample in enumerate(self.samples):
            x, y = sample.to_xy()

            xs.append(x)
            y_sample[i] = y

        X_sample = np.array(xs, dtype=np.float32)
        Y_sample = y_sample.reshape(-1, 1)

        return X_sample, Y_sample
