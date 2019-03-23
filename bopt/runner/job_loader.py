import numpy as np
from typing import List, Union, Type, Dict

from bopt.job_params import JobParams
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
    def from_dict(data: dict, hyperprameters: List[Hyperparameter]) -> Job:
        job_type = data["job_type"]

        if job_type not in JOB_MAPPING:
            raise NotImplemented()

        cls: JobTypes = JOB_MAPPING[data["job_type"]]

        # TODO: 64 or 32 bit?
        x = np.array(data["run_parameters"], dtype=np.float64)
        job_params = JobParams.mapping_from_vector(x, hyperprameters)

        job = cls(data["job_id"], job_params)
        job.started_at = data["started_at"]
        job.finished_at = data["finished_at"]

        return job

