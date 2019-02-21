import abc
import os
import yaml
import datetime

import numpy as np
from typing import Union, List, Optional, Tuple
from bopt.parsers.result_parser import ResultParser, JobResult

Timestamp = int
Value = float


class Job(abc.ABC):
    job_id: int
    run_parameters: dict
    started_at: datetime.datetime
    finished_at: datetime.datetime

    def to_dict(self) -> dict:
        return {
            "job_type": self.job_type(),
            "job_id": self.job_id,
            "run_parameters": self.run_parameters,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @abc.abstractmethod
    def job_type(self) -> str: pass

    @abc.abstractmethod
    def state(self): pass

    @abc.abstractmethod
    def kill(self) -> None: pass

    @abc.abstractmethod
    def is_finished(self) -> bool: pass

    # TODO: delete this
    def sorted_parameter_values(self) -> List[float]:
        # TODO: detect case when experiment results have a different number of params
        idx = np.argsort(list(self.run_parameters.keys()))
        return np.array(list(self.run_parameters.values()), dtype=np.float32)[idx]

    def get_result(self, output_dir: str) -> float:
        fname = os.path.join(output_dir, f"job-{self.job_id}.out")

        # TODO: handle errors
        with open(fname, "r") as f:
            return float(f.read())

    # def is_success(self) -> bool:
    #     if self.is_finished():
    #         return self.result_parser.safe_final_result(self).is_ok()
    #     else:
    #         return False
    #
    # def result_or_err(self) -> str:
    #     if self.is_finished():
    #         if self.is_success():
    #             return str(self.final_result())
    #         else:
    #             return self.err()
    #     else:
    #         return "RUNNING"
    #
    # def err(self) -> str:
    #     return self.result_parser.safe_final_result(self).err()
    #
    # def intermediate_results(self) -> List[float]:
    #     return self.result_parser.intermediate_results(self)
    #
    # def final_result(self) -> float:
    #     return self.result_parser.final_result(self)

    def status_str(self) -> str:
        if self.is_finished():
            return "FINISHED"
        else:
            return "RUNNING"
        # TODO: failed status


    def __str__(self) -> str:
        s = f"{self.job_id}\t"
        is_finished = self.is_finished()

        if self.is_finished():
            if self.is_success():
                final_result = self.final_result()

                rounded_params = {name: round(value, 4)
                        for name, value in self.run_parameters.items()}
                assert isinstance(final_result, float)
                s += f"{is_finished}\t{final_result:.3f}\t{rounded_params}"
            else:
                s += f"FAILED: {self.err()}"
        else:
            s += "RUNNING"

        return s

class Runner(abc.ABC):
    script_path: str
    arguments: List[str]

    @abc.abstractmethod
    def start(self, output_dir: str, run_parameters: dict) -> Job: pass

    @abc.abstractmethod
    def runner_type(self) -> str: pass

    def to_dict(self) -> dict:
        return {
            "runner_type": self.runner_type(),
            "script_path": self.script_path,
            "arguments": self.arguments,
        }

