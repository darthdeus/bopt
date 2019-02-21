import os
import datetime
import yaml
import re
import subprocess
import pathlib

from glob import glob
from typing import Union, List, Optional, Tuple

from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job, Runner, Timestamp, Value


class SGEJob(Job):
    def __init__(self, job_id: int, run_parameters: dict) -> None:
        self.job_id = job_id
        self.run_parameters = run_parameters
        self.started_at = None
        self.finished_at = None

    def job_type(self) -> str:
        return "sge_job"

    def is_finished(self) -> bool:
        raise NotImplementedError()

    def state(self) -> str:
        return subprocess.check_output(["qstat"]).decode("ascii")

    def kill(self):
        output = subprocess.check_output(["qdel", str(self.job_id)]).decode("ascii")

        assert re.match(pattern=".*has registered.*", string=output) is not None


class SGERunner(Runner):
    # TODO: share arguments in parent ctor
    def __init__(self, script_path: str, arguments: List[str]) -> None:
        self.script_path = script_path
        self.arguments = arguments

    def runner_type(self) -> str:
        return "sge_runner"

    def start(self, output_dir: str, run_parameters: dict) -> Job:
        run_params = [f"--{name}={value}" for name, value in run_parameters.items()]

        qsub_params: List[str] = ["-N", "job", "-o", output_dir]
        cmd = ["qsub", *qsub_params, self.script_path, *self.arguments, *run_params]

        print(f"Starting a new job: {' '.join(cmd)}")
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("ascii")
        print(output)

        matches = re.match(pattern=r"Your job (\d+) \(\".*\"\) has been submitted",
                           string=output)

        assert matches is not None
        job_id = int(matches.group(1))

        sge_job = SGEJob(job_id, run_parameters)
        sge_job.started_at = datetime.datetime.now()

        return sge_job
