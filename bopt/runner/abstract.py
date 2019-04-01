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

    def __init__(self, script_path: str, arguments: List[str]) -> None:
        self.script_path = script_path
        self.arguments = arguments

    # TODO: start by nemelo brat output_dir
    @abc.abstractmethod
    def start(self, output_dir: str, hyperparam_values: HyperparamValues) -> Job: pass

    @abc.abstractmethod
    def runner_type(self) -> str: pass

    def to_dict(self) -> dict:
        return {
            "runner_type": self.runner_type(),
            "script_path": self.script_path,
            "arguments": self.arguments,
        }

