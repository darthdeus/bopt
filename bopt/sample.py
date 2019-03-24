import os
import logging
import traceback
import numpy as np
from enum import Enum

from typing import Tuple, List, Optional

from bopt.basic_types import Hyperparameter
from bopt.hyperparam_values import HyperparamValues
from bopt.runner.abstract import Job
from bopt.runner.job_loader import JobLoader
from bopt.models.parameters import ModelParameters


class CollectFlag(Enum):
    WAITING_FOR_JOB = 1
    WAITING_FOR_SIMILAR = 2
    COLLECT_FAILED = 3
    COLLECT_OK = 4


class Sample:
    job: Optional[Job]
    model: ModelParameters

    hyperparam_values: HyperparamValues

    mu_pred: float
    sigma_pred: float
    collect_flag: CollectFlag

    result: Optional[float]
    comment: Optional[str]

    def __init__(self, job: Optional[Job],
            model_params: ModelParameters,
            hyperparam_values: HyperparamValues,
            mu_pred: float,
            sigma_pred: float,
            collect_flag: CollectFlag) -> None:
        self.job = job
        self.model = model_params
        self.hyperparam_values = hyperparam_values
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        self.result = None
        self.comment = None
        self.collect_flag = collect_flag

    def status(self) -> CollectFlag:
        return self.collect_flag

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict() if self.job else None,
            "model": self.model.to_dict() if self.model else None,
            "hyperparam_values": self.hyperparam_values.to_dict(),
            "result": self.result,
            "mu_pred": self.mu_pred,
            "sigma_pred": self.sigma_pred,
            "comment": self.comment,
            "collect_flag": self.collect_flag
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
                data["sigma_pred"],
                data["collect_flag"])

        sample.comment = data.get("comment", None)
        sample.result = data.get("result", None)

        return sample

    def to_x(self) -> np.ndarray:
        return self.hyperparam_values.x

    def to_xy(self) -> Tuple[np.ndarray, float]:
        x = self.to_x()

        status = self.status()

        assert self.result or self.job

        if status == CollectFlag.COLLECT_OK:
            y = self.result
        elif status == CollectFlag.WAITING_FOR_JOB or status == CollectFlag.WAITING_FOR_SIMILAR:
            y = self.mu_pred
        else:
            raise ValueError("Tried to get xy for a sample {} which is {}".format(self, status))

        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

        return x, y

    def __str__(self) -> str:
        if self.job:
            s = f"{self.job.job_id}\t"
        else:
            s = "manual\t"

        is_finished = self.status() == CollectFlag.COLLECT_OK

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
