import os
import pathlib
import yaml
import re
import numpy as np

from glob import glob
from typing import Union, NamedTuple, List

from myopt.runner.abstract import Job, Runner


class Integer:
    low: int
    high: int

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.type = "int"

    def sample(self) -> float:
        return np.random.randint(self.low, self.high)

    def __repr__(self) -> str:
        return f"Int({self.low}, {self.high})"

class Float:
    low: float
    high: float

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.type = "float"

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Float({self.low}, {self.high})"

Range = Union[Integer, Float]


class Hyperparameter(NamedTuple):
  name: str
  range: Range


META_FILENAME = "meta.yml"


class Experiment:
    meta_dir: str
    hyperparameters: List[Hyperparameter]
    runner: Runner
    evaluations: List[Job]

    def __init__(self, meta_dir: str, hyperparameters: List[Hyperparameter], runner: Runner):
        self.meta_dir = meta_dir
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.evaluations = []

        pathlib.Path(os.path.join(self.meta_dir, "outputs"))\
                .mkdir(parents=True, exist_ok=True)

        self.serialize()

    def iterate(self):
        eval_params = [param.sample() for param in self.hyperparameters]

        job = self.runner.start(eval_params)

        self.evaluations.append(job)

    @staticmethod
    def filename(directory) -> str:
        return os.path.join(directory, META_FILENAME)

    def serialize(self) -> None:
        evals = self.evaluations
        del self.evaluations
        dump = yaml.dump(self)
        self.evaluations = evals

        with open(Experiment.filename(self.meta_dir), "w") as f:
            f.write(dump)

        for job in self.evaluations:
            job.serialize()

    @staticmethod
    def deserialize(directory: str) -> "Experiment":
        with open(Experiment.filename(directory), "r") as f:
            contents = f.read()
            obj = yaml.load(contents)

        jobs = []
        for path in glob(os.path.join(directory, "job-*")):
            matches = re.match('.*?(\d+).*?', path)
            assert matches is not None

            job_id = int(matches.group(1))
            job = obj.runner.deserialize_job(obj.meta_dir, job_id)
            job.deserialize()

            jobs.append(job)
        obj.evaluations = jobs

        return obj
