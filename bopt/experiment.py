import numpy as np
import abc
import datetime

from typing import List

from bopt.kernels import Kernel
from bopt.runner.abstract import Runner


class Noise: pass
class Hyperparameter: pass


class Model:
    noise: Noise

class RandomSearch(Model):
    pass

class GP(Model):
    kernel: Kernel
    pass


class Job(abc.ABC):
    model: Model
    # run_parameters: dict


class MultiJob(Job):
    pass


class Sample:
    job: Job
    X: List[Hyperparameter]
    y: float
    other_params: dict

    started_at: datetime.datetime
    finished_at: datetime.datetime





class Experiment:
    hyperparameters: List[Hyperparameter]
    runner: Runner

    samples: List[Sample]



class Bopt:
    experiment: Experiment
    last_model: Model

