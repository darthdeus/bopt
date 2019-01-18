import abc
import os
import yaml

import numpy as np
from typing import Union, List, Optional, Tuple
from bopt.runner.result_parser import ResultParser, JobResult

Timestamp = int
Value = float


class Job(abc.ABC):
    meta_dir: str
    job_id: int
    result_parser: ResultParser
    run_parameters: dict

    @abc.abstractmethod
    def state(self): pass

    @abc.abstractmethod
    def kill(self) -> None: pass

    @abc.abstractmethod
    def is_finished(self) -> bool: pass

    def sorted_parameter_values(self) -> List[float]:
        # TODO: detect case when experiment results have a different number of params
        idx = np.argsort(list(self.run_parameters.keys()))
        return np.array(list(self.run_parameters.values()), dtype=np.float32)[idx]

    def is_success(self) -> bool:
        if self.is_finished():
            return self.result_parser.safe_final_result(self).is_ok()
        else:
            return False

    def result_or_err(self) -> str:
        if self.is_finished():
            if self.is_success():
                return str(self.final_result())
            else:
                return self.err()
        else:
            return "RUNNING"

    def err(self) -> str:
        return self.result_parser.safe_final_result(self).err()

    def intermediate_results(self) -> List[float]:
        return self.result_parser.intermediate_results(self)

    def final_result(self) -> float:
        return self.result_parser.final_result(self)

    def serialize(self) -> None:
        with open(self.filename(), "w") as f:
            f.write(yaml.dump(self))

    def deserialize(self) -> "Job":
        # raise "TODO: pridat hyperparam k jobu"
        with open(self.filename(), "r") as f:
            content = f.read()
            obj = yaml.load(content)

            assert self.meta_dir == obj.meta_dir
            assert self.job_id == obj.job_id

        return self

    def filename(self) -> str:
        return Job.compute_job_filename(self.meta_dir, self.job_id)

    def job_output_filename(self) -> str:
        return Job.compute_job_output_filename(self.meta_dir, self.job_id)

    def get_job_output(self) -> str:
        with open(self.job_output_filename(), "r") as f:
            return f.read().strip()

    @staticmethod
    def compute_job_filename(meta_dir, job_id) -> str:
        return os.path.join(meta_dir, f"job-{job_id}.yml")

    @staticmethod
    def compute_job_output_filename(meta_dir, job_id) -> str:
        return os.path.join(meta_dir, "outputs", f"job.o{job_id}")

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
    @abc.abstractmethod
    def start(self, run_parameters: dict) -> Job: pass

    @abc.abstractmethod
    def deserialize_job(self, meta_dir: str, job_id: int) -> Job: pass
