import abc
import os
import yaml
from typing import Union, List, Optional, Tuple
from bopt.runner.parser import ResultParser

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

    def final_result(self) -> Union[float, str]:
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

        if is_finished:
            try:
                final_result = self.final_result()

                rounded_params = {name: round(value, 4) for name, value in self.run_parameters.items()}
                s += f"{is_finished}\t{round(final_result, 3)}\t{rounded_params}"
            except ValueError as e:
                s += str(e)
        else:
            s += "RUNNING"

        return s

class Runner(abc.ABC):
    @abc.abstractmethod
    def start(self, run_parameters: dict) -> Job: pass

    @abc.abstractmethod
    def deserialize_job(self, meta_dir: str, job_id: int) -> Job: pass
