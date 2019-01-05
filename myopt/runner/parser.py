import abc
from typing import List, Union


class ResultParser(abc.ABC):
    @abc.abstractmethod
    def intermediate_results(self, job) -> List[float]: pass

    @abc.abstractmethod
    def final_result(self, job) -> Union[float, str]: pass


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class LastLineLastWordParser(ResultParser):
    """
    Parses the last word on the last line in the output file
    as a float.
    """

    def intermediate_results(self, job) -> List[float]:
        # Skip last line
        lines = job.get_job_output().split("\n")[:-1]

        last_words = [line.split(" ")[-1] for line in lines]

        return [float(word) for word in last_words if is_float(word)]

    def final_result(self, job) -> Union[float, str]:
        output = job.get_job_output()
        last_line = output.split("\n")[-1].strip()
        last_word = last_line.split(" ")[-1]
        try:
            return float(last_word)
        except ValueError:
            return f"Last line not ending with float: '{last_line}'"

