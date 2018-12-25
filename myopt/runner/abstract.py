import abc
import os
import yaml
from typing import Union, List, Optional, Tuple

Timestamp = int
Value = float


class Job(abc.ABC):
    meta_dir: str
    job_id: int
    is_finished: bool
    intermediate_results : List[Tuple[Timestamp, Value]]
    final_result: Optional[Value]

    @abc.abstractmethod
    def state(self): pass

    @abc.abstractmethod
    def kill(self) -> None: pass

    def serialize(self) -> None:
        with open(self.filename(), "w") as f:
            f.write(yaml.dump(self))

    def deserialize(self) -> "Job":
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
        return os.path.join(self.meta_dir, f"job-{self.job_id}.yml")

    def job_output_filename(self) -> str:
        return Job.compute_job_output_filename(self.meta_dir, self.job_id)

    def get_job_output(self) -> str:
        with open(self.job_output_filename(), "r") as f:
            return f.read()

    @staticmethod
    def compute_job_output_filename(meta_dir, job_id):
        return os.path.join(meta_dir, "outputs", f"job.o{job_id}")

class Runner(abc.ABC):
    @abc.abstractmethod
    def start(self, run_parameters: dict) -> Job: pass

    @abc.abstractmethod
    def deserialize_job(self, meta_dir: str, job_id: int) -> Job: pass
