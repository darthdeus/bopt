import abc
import os
import yaml
import datetime
import logging
import traceback
import re

import numpy as np
from typing import Union, List, Optional, Tuple
from bopt.hyperparam_values import HyperparamValues

Timestamp = int
Value = float


class Job(abc.ABC):
    job_id: int

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id

    def to_dict(self) -> dict:
        return {
            "job_type": self.job_type(),
            "job_id": self.job_id,
        }

    @abc.abstractmethod
    def job_type(self) -> str: pass

    @abc.abstractmethod
    def kill(self) -> None: pass

    @abc.abstractmethod
    def is_finished(self) -> bool: pass


class Runner(abc.ABC):
    script_path: str
    arguments: List[str]
    manual_arg_fnames: List[str]

    def __init__(self, script_path: str, arguments: List[str],
            manual_arg_fnames: List[str]) -> None:
        self.script_path = script_path
        self.arguments = arguments
        self.manual_arg_fnames = manual_arg_fnames

    # TODO: start by nemelo brat output_dir
    @abc.abstractmethod
    def start(self, output_dir: str,
              hyperparam_values: HyperparamValues,
              manual_file_args: List[str]) -> Job: pass

    @abc.abstractmethod
    def runner_type(self) -> str: pass

    def to_dict(self) -> dict:
        return {
            "runner_type": self.runner_type(),
            "script_path": self.script_path,
            "arguments": self.arguments,
            "manual_arg_fnames": self.manual_arg_fnames,
        }

    def fetch_and_shift_manual_file_args(self) -> List[str]:
        result = []

        for fname in self.manual_arg_fnames:
            with open(fname, "r") as fin:
                data = fin.read().splitlines(True)
            with open(fname, "w") as fout:
                fout.writelines(data[1:])

            value = data[0].strip()
            param_name = os.path.splitext(os.path.basename(fname))[0]

            result.append("--{}={}".format(param_name, value))

        return result
