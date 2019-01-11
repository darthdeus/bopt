import abc
from typing import List, Union, Generic, TypeVar, cast

T = TypeVar("T")
E = TypeVar("E")


class Result(Generic[T, E]):
    def __init__(self, is_ok: bool, value: Union[T, E]) -> None:
        self._is_ok = is_ok
        self._value = value

    def is_ok(self) -> bool:
        return self._is_ok

    def is_err(self) -> bool:
        return not self._is_ok

    def value(self) -> T:
        assert self._is_ok
        return cast(T, self._value)

    def err(self) -> E:
        assert not self._is_ok
        return cast(E, self._value)

    @staticmethod
    def Ok(value) -> "Result":
        return Result(True, value)

    @staticmethod
    def Err(value) -> "Result":
        return Result(False, value)


JobResult = Result[float, str]


class ResultParser(abc.ABC):
    @abc.abstractmethod
    def intermediate_results(self, job) -> List[float]: pass

    @abc.abstractmethod
    def safe_final_result(self, job) -> JobResult: pass

    @abc.abstractmethod
    def final_result(self, job) -> float: pass

    @abc.abstractmethod
    def has_final_result(self, job) -> bool: pass


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

    def safe_final_result(self, job) -> JobResult:
        output = job.get_job_output()
        last_line = output.split("\n")[-1].strip()
        last_word = last_line.split(" ")[-1]
        try:
            return Result.Ok(float(last_word))
        except ValueError:
            return Result.Err(f"Last line not ending with float: '{last_line}'")

    def final_result(self, job) -> float:
        result = self.safe_final_result(job)
        assert result.is_ok

        return result.value()

    def has_final_result(self, job) -> bool:
        result = self.safe_final_result(job)

        return result.is_ok()

