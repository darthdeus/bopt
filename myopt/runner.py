import abc
import os
import yaml
import re
import subprocess

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
    def state(self): pass

    @abc.abstractmethod
    def kill(self) -> None: pass

    @abc.abstractmethod
    def serialize(self) -> None: pass

    @abc.abstractmethod
    def deserialize(self): pass


class Runner(abc.ABC):
    @abc.abstractmethod
    def start(self, run_parameters: dict) -> Job: pass


class SGEJob(Job):
    def __init__(self, meta_dir: str, job_id: int) -> None:
        self.meta_dir = meta_dir
        self.job_id = job_id

        self.is_finished = False
        self.intermediate_results = []
        self.final_result = None

        self.serialize()


    is_finished: bool
    intermediate_results : List[Tuple[Timestamp, Value]]
    final_result: Optional[Value]

    def state(self) -> str:
        return subprocess.check_output(["qstat"])

    def kill(self):
        output = subprocess.check_output(["qdel", str(self.job_id)])

        assert re.match(pattern=".*has registered.*", string=output) is not None

    def serialize(self) -> None:
        with open(self.filename(), "w") as f:
            f.write(yaml.dump(self))

    def deserialize(self) -> None:
        with open(self.filename(), "r") as f:
            content = f.read()
            obj = yaml.load(content)

            assert self.meta_dir == obj.dir
            assert self.job_id == obj.job_id

            self.is_finished = obj.is_finished
            self.intermediate_results = obj.intermediate_results
            self.final_result = obj.final_result

    def filename(self) -> str:
        return os.path.join(self.meta_dir, str(self.job_id), ".yml")


QSUB_JOBID_PATTERN = "Your job (\d+) \(\".*\"\) has been submitted"


class SGERunner(Runner):
    meta_dir: str
    script_path: str
    arguments: List[str]

    def __init__(self, meta_dir: str, script_path: str, arguments: List[str]) -> None:
        self.script_path = script_path
        self.arguments = arguments
        self.meta_dir = meta_dir

    def start(self, run_parameters: dict) -> Job:
        run_params = [f"--{name}={value}" for name, value in run_parameters.items()]

        cmd = [self.script_path, *self.arguments, *run_params]

        output = subprocess.check_output(["qsub", *cmd], stderr=subprocess.STDOUT)

        matches = re.match(pattern=QSUB_JOBID_PATTERN, string=output)

        assert matches is not None

        job_id = int(matches.group(0))

        return SGEJob(self.meta_dir, job_id)


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

