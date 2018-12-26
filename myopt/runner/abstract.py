import abc
import os
import yaml
from typing import Union, List, Optional, Tuple
from myopt.runner.parser import ResultParser

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

            self.is_finished = obj.is_finished
            self.intermediate_results = obj.intermediate_results
            self.final_result = obj.final_result

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

class Runner(abc.ABC):
    @abc.abstractmethod
    def start(self, run_parameters: dict) -> Job: pass

    @abc.abstractmethod
    def deserialize_job(self, meta_dir: str, job_id: int) -> Job: pass
