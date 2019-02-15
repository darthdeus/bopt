import datetime

from typing import Optional, List

from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job
from bopt.gaussian_process import GaussianProcess


class Sample:
    model: GaussianProcess

    hyperparam_values: List[Hyperparameter]
    run_params: dict

    job: Job

