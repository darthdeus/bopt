import abc


class ResultParser(abc.ABC):
    @abc.abstractmethod
    def __call__(self, job) -> float:
        pass


class LastLineLastWordParser(ResultParser):
    """
    Parses the last word on the last line in the output file
    as a float.
    """
    def __call__(self, job) -> float:
        return float(job.get_job_output().split("\n")[-1].split(" ")[-1])

