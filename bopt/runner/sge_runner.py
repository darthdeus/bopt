import os
import re
import subprocess
import logging

from typing import List

from bopt.hyperparam_values import HyperparamValues
from bopt.runner.abstract import Job, Runner


class SGEJob(Job):
    def job_type(self) -> str:
        return "sge_job"

    def is_finished(self) -> bool:
        try:
            fnull = open(os.devnull, "w")
            subprocess.check_output(["qstat", "-j", str(self.job_id)], stderr=fnull)
            return False
        except subprocess.CalledProcessError:
            return True

    def kill(self):
        try:
            output = subprocess.check_output(["qdel", str(self.job_id)]).decode("ascii")

            if not re.match(pattern=".*has registered.*", string=output):
                logging.error("qdel returned unexpected output: {}".format(output))
        except subprocess.CalledProcessError as e:
            logging.error("Failed to kill a job {} with error {}".format(self.job_id, e))


class SGERunner(Runner):
    qsub_arguments: List[str]

    def __init__(self, script_path: str, arguments: List[str],
                 qsub_arguments: List[str],
                 manual_arg_fnames: List[str]) -> None:
        super().__init__(script_path, arguments, manual_arg_fnames)
        self.qsub_arguments = qsub_arguments

    def runner_type(self) -> str:
        return "sge_runner"

    def start(self, output_dir: str, hyperparam_values: HyperparamValues,
              manual_file_args: List[str]) -> Job:
        cmdline_run_args = [f"--{h.name}={value}"
                            for h, value in hyperparam_values.mapping.items()]

        override_qsub_args = os.environ.get("QSUB_PARAMS")

        if override_qsub_args:
            additional_args = override_qsub_args.split(":::")
            logging.info("Overriding {} with {}".format(self.qsub_arguments, additional_args))
        else:
            additional_args = self.qsub_arguments

        qsub_params = ["-N", "job", "-o", output_dir, *additional_args]
        cmd = [
            "qsub",
            *qsub_params,
            self.script_path,
            *self.arguments,
            *manual_file_args,
            *cmdline_run_args
        ]

        logging.info(f"SGE_JOB_START: {' '.join(cmd)}")
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("ascii")
        except subprocess.CalledProcessError as e:
            print("Command failed, output:\n{}".format(e.output))
            raise e

        matches = re.match(
                pattern=r"Your job (\d+) \(\".*\"\) has been submitted",
                string=output)

        assert matches is not None
        job_id = int(matches.group(1))

        return SGEJob(job_id)

    def to_dict(self) -> dict:
        return {
            "runner_type": self.runner_type(),
            "script_path": self.script_path,
            "arguments": self.arguments,
            "qsub_arguments": self.qsub_arguments,
            "manual_arg_fnames": self.manual_arg_fnames,
        }
