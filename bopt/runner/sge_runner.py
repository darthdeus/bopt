import os
import yaml
import re
import subprocess
import pathlib

from glob import glob
from typing import Union, List, Optional, Tuple

from bopt.hyperparameters import Hyperparameter
from bopt.runner.abstract import Job, Runner, Timestamp, Value


class SGEJob(Job):
    is_finished: bool
    intermediate_results : List[Tuple[Timestamp, Value]]
    final_result: Optional[Value]

    def __init__(self, meta_dir: str, job_id: int) -> None:
        self.meta_dir = meta_dir
        self.job_id = job_id

        self.is_finished = False
        self.intermediate_results = []
        self.final_result = None

        self.serialize()

    def state(self) -> str:
        return subprocess.check_output(["qstat"]).decode("ascii")

    def kill(self):
        output = subprocess.check_output(["qdel", str(self.job_id)]).decode("ascii")

        assert re.match(pattern=".*has registered.*", string=output) is not None


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

        output_dir = os.path.join(self.meta_dir, "outputs")
        qsub_params: List[str] = ["-N", "job", "-o", output_dir]
        cmd = ["qsub", *qsub_params, self.script_path, *self.arguments, *run_params]

        print(f"Starting a new job: {' '.join(cmd)}")
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("ascii")
        print(output)

        matches = re.match(pattern=QSUB_JOBID_PATTERN, string=output)

        assert matches is not None
        job_id = int(matches.group(1))

        return SGEJob(self.meta_dir, job_id)

    def deserialize_job(self, meta_dir: str, job_id: int) -> Job:
        return SGEJob(meta_dir, job_id)
