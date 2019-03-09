import abc
import numpy as np
from typing import Union, NamedTuple, List


# TODO: fix naming convention & typnig errors
class Bound(abc.ABC):
    pass


class Integer(Bound):
    low: int
    high: int
    type: str

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.type = "int"

    def sample(self) -> float:
        return np.random.randint(self.low, self.high)

    def __repr__(self) -> str:
        return f"Int({self.low}, {self.high})"


class Float(Bound):
    low: float
    high: float
    type: str

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.type = "float"

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Float({self.low}, {self.high})"


class Discrete(Bound):
    low: float
    high: float
    type: str

    def __init__(self, values: List[str]):
        self.values = values
        self.type = "discrete"
        self.low = 0
        self.high = len(values)

    def sample(self) -> float:
        return np.random.randint(self.low, self.high)

    def __repr__(self) -> str:
        return f"Discrete({self.values})"


class Hyperparameter(NamedTuple):
    name: str
    range: Bound

    def to_dict(self) -> dict:
        if self.range.type == "discrete":
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

    @staticmethod
    def from_dict(name, data: dict) -> "Hyperparameter":
        if data["type"] == "discrete":
            return Hyperparameter(name=name,
                    range=Discrete(data["values"]))
        else:
            if data["type"] == "int":
                cls = Integer
                parser = int
            elif data["type"] == "float":
                cls = Float
                parser = float
            else:
                raise NotImplementedError()

            return Hyperparameter(name=name,
                    range=cls(parser(data["low"]), parser(data["high"])))
