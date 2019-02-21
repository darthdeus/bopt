from bopt.runner.abstract import Job
from bopt.runner.local_runner import LocalJob
from bopt.runner.sge_runner import SGEJob

JOB_MAPPING = {
    "local_job": LocalJob,
    "sge_job": SGEJob,
}

class JobLoader:

    @staticmethod
    def from_dict(self, data: dict) -> Job:
        job_type = data["job_type"]

        if job_type not in JOB_MAPPING:
            raise NotImplemented()

        cls = JOB_MAPPING[data["job_type"]]

        job = cls(data["job_id"], data["run_parameters"])
        job.started_at = data["started_at"]
        job.finished_at = data["finished_at"]

        return job

