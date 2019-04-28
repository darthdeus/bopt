import datetime
import math
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

    mu_pred: Optional[float]
    sigma_pred: Optional[float]
    collect_flag: CollectFlag

    result: Optional[float]
    comment: Optional[str]

    created_at: datetime.datetime
    finished_at: Optional[datetime.datetime]
    collected_at: Optional[datetime.datetime]
    run_time: Optional[float]   # in seconds

    def __init__(self, job: Optional[Job],
            model_params: ModelParameters,
            hyperparam_values: HyperparamValues,
            mu_pred: Optional[float],
            sigma_pred: Optional[float],
            collect_flag: CollectFlag,
            created_at: datetime.datetime) -> None:
        self.job = job
        self.model = model_params

        self.hyperparam_values = hyperparam_values

        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        self.collect_flag = collect_flag

        self.result = None
        self.comment = None

        self.created_at = created_at
        self.finished_at = None
        self.collected_at = None
        self.run_time = None

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
            "collect_flag": self.collect_flag,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "collected_at": self.collected_at,
            "run_time": self.run_time
        }

    @staticmethod
    def from_dict(data: dict, hyperparameters: List[Hyperparameter]) -> "Sample":
        model_dict = ModelParameters.from_dict(data["model"])

        # TODO: 64 or 32 bit?
        x = np.array(data["hyperparam_values"], dtype=np.float64)
        hyperparam_values = HyperparamValues.mapping_from_vector(x, hyperparameters)

        job: Optional[Job]

        if "job" in data and data["job"] is not None:
            job = JobLoader.from_dict(data["job"])
        else:
            job = None

        sample = Sample(job,
                model_dict,
                hyperparam_values,
                data["mu_pred"],
                data["sigma_pred"],
                data["collect_flag"],
                data["created_at"])

        sample.comment = data.get("comment", None)
        sample.result = data.get("result", None)
        sample.collected_at = data.get("collected_at", None)
        sample.finished_at = data.get("finished_at", None)
        sample.run_time = data.get("run_time", None)

        return sample

    def to_x(self) -> np.ndarray:
        return self.hyperparam_values.x

    def to_xy(self) -> Tuple[np.ndarray, float]:
        x = self.to_x()

        status = self.status()

        assert self.result or self.job or status == CollectFlag.WAITING_FOR_SIMILAR, \
            "status was: {}".format(status)

        if status == CollectFlag.COLLECT_OK:
            y = self.result
        elif status == CollectFlag.WAITING_FOR_JOB \
                or status == CollectFlag.WAITING_FOR_SIMILAR:
            assert self.mu_pred, "No mu_pred for {}".format(self)

            y = self.mu_pred
        elif status == CollectFlag.COLLECT_FAILED:
            assert self.mu_pred
            assert self.sigma_pred

            y = self.mu_pred - self.sigma_pred
        else:
            raise ValueError("Tried to get xy for a sample {} which is {}".format(self, status))

        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

        assert not math.isnan(y)

        return x, y

    def get_output(self):
        # TODO: duplicitni
        fname = os.path.join("output", f"job.o{self.job.job_id}")
        print("Trying to get output of {}".format(os.path.abspath(fname)))
        if os.path.exists(fname):
            with open(fname, "r") as f:
                contents = f.read().strip()[-100000:]
                return contents
        else:
            return None


    def __str__(self) -> str:
        if self.job:
            s = f"{self.job.job_id}\t"
        else:
            s = "manual\t"

        s += "s: {}\tf: {}\tc: {}\tr: {}\t".format(self.created_at,
                self.finished_at, self.collected_at, self.run_time)

        s += "RS: {}".format(self.model.sampled_from_random_search())

        is_finished = self.status() == CollectFlag.COLLECT_OK

        # TODO: proper status check

        # TODO: hyperparam values __str__ a pouzivat to i jinde
        if self.result:
            rounded_params = {h.name: (round(v, 3) if isinstance(v, float) else v)
                    for h, v in self.hyperparam_values.mapping.items()}

            assert isinstance(self.result, float)
            s += f"{is_finished}\t{self.result:.3f}\t{rounded_params}"
        else:
            s += str(self.status())

        return s

    def short_collect_flag_str(self) -> str:
        return str(self.collect_flag)\
                .replace("CollectFlag.", "")\
                .replace("COLLECT_", "")\
                .replace("WAITING_FOR_", "W_")

    def run_time_str(self) -> str:
        if self.run_time:
            time = int(self.run_time)

            if time > 3600:
                hours = time // 3600
                return "{}:{}".format(hours, (time - hours * 3600) // 60)
            elif time > 60:
                minutes = time // 60
                return "{}min".format(minutes)
            else:
                return "{}.s".format(time)
        else:
            return ""


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
