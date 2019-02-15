import numpy as np
import abc
import datetime

from typing import List, Optional

from bopt.kernels import Kernel
from bopt.runner.abstract import Runner


# class Noise: pass
# class Hyperparameter: pass
#
#
# class Model:
#     noise: Noise
#
#
# class RandomSearch(Model):
#     pass
#
#
# class GP(Model):
#     kernel: Kernel
#     pass
#
#
# class Job(abc.ABC):
#     model: Model
#     # run_parameters: dict
#
#
# class MultiJob(Job):
#     pass
#
#
# class Sample:
#     hyperparameters: List[Hyperparameter]
#     other_params: dict
#
#     started_at: datetime.datetime
#     finished_at: datetime.datetime
#
#     job: Optional[Job]
#
#     def y(self) -> Optional[float]:
#         pass
#
#     X: List[Hyperparameter]
#     y: float
#
#
#
#
# class Experiment:
#     hyperparameters: List[Hyperparameter]
#     runner: Runner
#     samples: List[Sample]
#     last_model: Model



# import bopt
#
# hyperparameters = [
#     bopt.Hyperparameter("gamma", bopt.Float(0, 1)),
#     bopt.Hyperparameter("epsilon", bopt.Float(0, 1)),
# ]
#
# runner = bopt.LocalRunner(
#         "./.venv/bin/python",
#         ["./experiments/rl/monte_carlo.py"],
#         bopt.LastLineLastWordParser())
#
# experiment = bopt.Experiment(hyperparameters, runner)
#
# meta_dir = "results/rl-monte-carlo"
# experiment.run(meta_dir, n_iter=20)
