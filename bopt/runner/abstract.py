import abc
import os
import yaml
import datetime
import logging
import traceback
import re

import numpy as np
from typing import Union, List, Optional, Tuple
from bopt.job_params import JobParams

Timestamp = int
Value = float


class Job(abc.ABC):
    job_id: int
    run_parameters: JobParams
    started_at: Optional[datetime.datetime]
    finished_at: Optional[datetime.datetime]

    def __init__(self, job_id: int, run_parameters: JobParams) -> None:
        assert isinstance(run_parameters, JobParams)
        self.job_id = job_id
        self.run_parameters = run_parameters
        self.started_at = None
        self.finished_at = None

    def to_dict(self) -> dict:
        return {
            "job_type": self.job_type(),
            "job_id": self.job_id,
            "run_parameters": self.run_parameters.to_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
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

    @abc.abstractmethod
    def start(self, output_dir: str, run_parameters: JobParams) -> Job: pass

    @abc.abstractmethod
    def runner_type(self) -> str: pass

    def to_dict(self) -> dict:
        return {
            "runner_type": self.runner_type(),
            "script_path": self.script_path,
            "arguments": self.arguments,
        }

