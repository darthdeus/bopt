import abc
import numpy as np
from enum import Enum
from typing import Union, NamedTuple, List, Tuple, Type


ParamTypes = Union[float, int, str]


class OptimizationFailed(Exception):
    pass


class JobStatus(Enum):
    QUEUED = 1
    RUNNING = 2
    FAILED = 3
    CANCELED = 4
    FINISHED = 5


# TODO: fix naming convention & typnig errors
class Bound(abc.ABC):
    low: float
    high: float
    type: str

    @abc.abstractmethod
    def is_discrete(self) -> bool:
        pass

    @abc.abstractmethod
    def sample(self) -> ParamTypes:
        pass

    @abc.abstractmethod
    def validate(self, value: ParamTypes) -> bool:
        pass

    @abc.abstractmethod
    def parse(self, value: str) -> ParamTypes:
        return value

    @abc.abstractmethod
    def scipy_bound_tuple(self) -> Tuple[float, float]:
        pass


class Integer(Bound):
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.type = "int"

    def sample(self) -> float:
        return np.random.randint(self.low, self.high)

    def is_discrete(self) -> bool:
        return True

    def validate(self, value: ParamTypes) -> bool:
        assert isinstance(value, int)
        return self.low <= value < self.high

    def __repr__(self) -> str:
        return f"Int({self.low}, {self.high})"

    def parse(self, value: str) -> ParamTypes:
        return int(value)

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        return (self.low, (self.high - 1))


class Float(Bound):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.type = "float"

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

    def is_discrete(self) -> bool:
        return False

    def validate(self, value: ParamTypes) -> bool:
        assert isinstance(value, float)
        return self.low <= value < self.high

    def __repr__(self) -> str:
        return f"Float({self.low}, {self.high})"

    def parse(self, value: str) -> ParamTypes:
        return float(value)

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        return (self.low, self.high - 1e-8)


class Discrete(Bound):
    def __init__(self, values: List[str]):
        self.values = values
        self.type = "discrete"
        self.low = 0
        self.high = len(values)

    def sample(self) -> float:
        return self.values[np.random.randint(self.low, self.high)]

    def validate(self, value: ParamTypes) -> bool:
        assert isinstance(value, str)
        return value in self.values

    def is_discrete(self) -> bool:
        return True

    def map(self, value) -> int:
        return self.values.index(value)

    def inverse_map(self, value) -> str:
        return self.values[value]

    def __repr__(self) -> str:
        return f"Discrete({self.values})"

    def parse(self, value: str) -> ParamTypes:
        return value

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        return (self.low, (self.high - 1))


class Hyperparameter(NamedTuple):
    name: str
    range: Bound

    def to_dict(self) -> dict:
        if isinstance(self.range, Discrete):
            return {
                "type": "discrete",
                "values": self.range.values
            }
        else:
            return {
                "type": self.range.type,
                "low": self.range.low,
                "high": self.range.high
            }

    def validate(self, value) -> bool:
        return self.range.validate(value)

    @staticmethod
    def from_dict(name, data: dict) -> "Hyperparameter":
        if data["type"] == "discrete":
            return Hyperparameter(name=name,
                    range=Discrete(data["values"]))
        elif data["type"] == "int":
            return Hyperparameter(name=name,
                    range=Integer(int(data["low"]), int(data["high"])))
        elif data["type"] == "float":
            return Hyperparameter(name=name,
                    range=Float(float(data["low"]), float(data["high"])))
        else:
            raise NotImplementedError()
