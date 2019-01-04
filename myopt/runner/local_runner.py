import os
import yaml
import re
import subprocess
import pathlib
import psutil
import tempfile

from glob import glob
from typing import Union, List, Optional, Tuple, Callable

from myopt.hyperparameters import Hyperparameter
from myopt.runner.abstract import Job, Runner, Timestamp, Value
from myopt.runner.parser import ResultParser


class LocalJob(Job):
    def __init__(self, meta_dir: str, job_id: int, result_parser: ResultParser,
                 run_parameters: dict) -> None:
        self.meta_dir = meta_dir
        self.job_id = job_id
        self.result_parser = result_parser
        self.run_parameters = run_parameters

        self.serialize()

    def is_finished(self) -> bool:
        return not psutil.pid_exists(self.job_id)

    def state(self) -> bool:
        pass

    def kill(self) -> None:
        if psutil.pid_exists(self.job_id):
            psutil.Process(self.job_id).kill()


class LocalRunner(Runner):
    hidden_fields = ["result_parser"]

    meta_dir: str
    script_path: str
    arguments: List[str]

    def __init__(self, meta_dir: str, script_path: str, arguments: List[str],
                 result_parser: ResultParser) -> None:
        self.script_path = script_path
        self.arguments = arguments
        self.meta_dir = meta_dir
        self.result_parser = result_parser

    def start(self, run_parameters: dict) -> Job:
        cmdline_run_params = [f"--{name}={value}" for name, value in run_parameters.items()]

        output_dir = os.path.join(self.meta_dir, "outputs")
        cmd = [self.script_path, *self.arguments, *cmdline_run_params]

        print(f"Starting a new job: {' '.join(cmd)}")

        temp_fname = tempfile.mktemp(dir=".")
        with open(temp_fname, "w") as f:
            process = psutil.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

            job_id = process.pid
            job_fname = Job.compute_job_output_filename(self.meta_dir, job_id)

            os.rename(temp_fname, job_fname)

            return LocalJob(self.meta_dir, job_id, self.result_parser, run_parameters)

    def deserialize_job(self, meta_dir: str, job_id: int) -> Job:
        fname = Job.compute_job_filename(meta_dir, job_id)

        with open(fname, "r") as f:
            return yaml.load(f.read())
