import datetime
import yaml
import re
import os
import subprocess
import pathlib
import psutil
import tempfile
import logging

from glob import glob
from typing import Union, List, Optional, Tuple, Callable

from bopt.hyperparam_values import HyperparamValues
from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job, Runner, Timestamp, Value


class LocalJob(Job):
    def job_type(self) -> str:
        return "local_job"

    def is_finished(self) -> bool:
        return not psutil.pid_exists(self.job_id)

    def kill(self) -> None:
        if psutil.pid_exists(self.job_id):
            psutil.Process(self.job_id).kill()


class LocalRunner(Runner):
    def runner_type(self) -> str:
        return "local_runner"

    def start(self, output_dir: str, hyperparam_values: HyperparamValues, manual_file_args: List[str]) -> Job:
        cmdline_run_params = [f"--{h.name}={value}"
                for h, value in hyperparam_values.mapping.items()]

        cmd = list(map(lambda x: os.path.expanduser(x), [
            self.script_path,
            *self.arguments,
            *manual_file_args,
            *cmdline_run_params
        ]))

        temp_fname = tempfile.mktemp(dir=output_dir)
        with open(temp_fname, "w") as f:
            process = psutil.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

            job_id = process.pid
            job_fname = os.path.join(output_dir, f"job.o{job_id}")

            # TODO: zkontrolovat duplicitni job_id, neudelat rename, smazat tempfile

            logging.info(f"JOB_START {job_id}, params:\n{hyperparam_values}") # :\t{' '.join(cmd)}")

            os.rename(temp_fname, job_fname)

            return LocalJob(job_id)
