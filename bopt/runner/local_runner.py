import datetime
import yaml
import re
import os
import subprocess
import pathlib
import psutil
import tempfile

from glob import glob
from typing import Union, List, Optional, Tuple, Callable

from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job, Runner, Timestamp, Value


class LocalJob(Job):
    def job_type(self) -> str:
        return "local_job"

    def is_finished(self) -> bool:
        return not psutil.pid_exists(self.job_id)

    def state(self) -> bool:
        pass

    def kill(self) -> None:
        if psutil.pid_exists(self.job_id):
            psutil.Process(self.job_id).kill()


class LocalRunner(Runner):
    def runner_type(self) -> str:
        return "local_runner"

    def start(self, output_dir: str, run_parameters: dict) -> Job:
        cmdline_run_params = [f"--{name}={value}" for name, value in run_parameters.items()]

        cmd = list(map(lambda x: os.path.expanduser(x), [
            self.script_path,
            *self.arguments,
            *cmdline_run_params
        ]))

        temp_fname = tempfile.mktemp(dir=output_dir)
        with open(temp_fname, "w") as f:
            process = psutil.Popen(cmd, stdout=f)
            # TODO: stderr?
            # , stderr=subprocess.STDOUT)

            job_id = process.pid
            job_fname = os.path.join(output_dir, f"job-{job_id}.out")

            print(f"START {job_id}") # :\t{' '.join(cmd)}")

            os.rename(temp_fname, job_fname)

            local_job = LocalJob(job_id, run_parameters)
            local_job.started_at = datetime.datetime.now()

            return local_job
