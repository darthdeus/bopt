import os
import datetime
import yaml
import re
import subprocess
import pathlib
import logging

from glob import glob
from typing import Union, List, Optional, Tuple

from bopt.job_params import JobParams
from bopt.basic_types import Hyperparameter, JobStatus
from bopt.runner.abstract import Job, Runner, Timestamp, Value


class SGEJob(Job):
    def job_type(self) -> str:
        return "sge_job"

    def is_finished(self) -> bool:
        try:
            subprocess.check_output(["qstat", "-j", str(self.job_id)])
            return False
        except subprocess.CalledProcessError:
            return True

    def status(self) -> JobStatus:
        if not self.is_finished():
            return JobStatus.RUNNING
        else:
            try:
                result = self.get_result()
                return JobStatus.FINISHED
            except ValueError as e:
                logging.error("Failed to parse result {}".format(e))
                return JobStatus.FAILED

    # def state(self) -> str:
    #     return subprocess.check_output(["qstat"]).decode("ascii")

    def kill(self):
        output = subprocess.check_output(["qdel", str(self.job_id)]).decode("ascii")

        assert re.match(pattern=".*has registered.*", string=output) is not None


class SGERunner(Runner):
    def runner_type(self) -> str:
        return "sge_runner"

    def start(self, output_dir: str, run_parameters: JobParams) -> Job:
        cmdline_run_params = [f"--{h.name}={value}" for h, value in run_parameters.mapping.items()]

        qsub_params: List[str] = ["-N", "job", "-o", output_dir]
        cmd = ["qsub", *qsub_params, self.script_path, *self.arguments, *cmdline_run_params]

        logging.info(f"SGE_JOB_START: {' '.join(cmd)}")
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("ascii")
        logging.info(output)

        matches = re.match(
                pattern=r"Your job (\d+) \(\".*\"\) has been submitted",
                string=output)

        assert matches is not None
        job_id = int(matches.group(1))

        sge_job = SGEJob(job_id, run_parameters)
        sge_job.started_at = datetime.datetime.now()

        return sge_job
