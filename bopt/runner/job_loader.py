import numpy as np
from typing import List, Union, Type, Dict

from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job
from bopt.runner.local_runner import LocalJob
from bopt.runner.sge_runner import SGEJob

JobTypes = Union[Type[LocalJob], Type[SGEJob]]


JOB_MAPPING: Dict[str, JobTypes] = {
    "local_job": LocalJob,
    "sge_job": SGEJob,
}


class JobLoader:
    @staticmethod
    def from_dict(data: dict) -> Job:
        job_type = data["job_type"]

        if job_type not in JOB_MAPPING:
            raise NotImplementedError()

        cls: JobTypes = JOB_MAPPING[data["job_type"]]

        return cls(data["job_id"])

