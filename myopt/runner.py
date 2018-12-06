import abc
import os
import yaml

from typing import Union, List, Optional, Tuple
from myopt.hyperparameters import Hyperparameter


Timestamp = int
Value = float


class Job(abc.ABC):
    # is_finished:
    is_finished: bool
    intermediate_results : List[Tuple[Timestamp, Value]]
    final_result: Optional[Value]

    @abc.abstractmethod
    def state(): pass

    @abc.abstractmethod
    def kill(): pass

    @abc.abstractmethod
    def serialize(): pass

    @abc.abstractmethod
    def deserialize(): pass


class Runner(abc.ABC):
    @abc.abstractmethod
    def start(*args, **kwargs) -> Job:
        pass


class SGERunner(Runner):
    script_path: str
    arguments: List[str]

    def __init__(self, script_path: str, arguments: List[str]) -> None:
        self.script_path = script_path
        self.arguments = arguments


META_FILENAME = "meta.yml"


class Experiment:
    hyperparameters: List[Hyperparameter]
    runner: Runner
    evaluations: List[Job]

    def __init__(self, hyperparameters: List[Hyperparameter], runner: Runner):
        self.hyperparameters = hyperparameters
        self.runner = runner

    def serialize(self, directory: str) -> None:
        dump = yaml.dump(self)

        with open(os.path.join(directory, META_FILENAME), "w") as f:
            f.write(dump)

    @staticmethod
    def deserialize(directory: str) -> "Experiment":
        with open(os.path.join(directory, META_FILENAME), "r") as f:
            contents = f.read()
            return yaml.load(contents)

