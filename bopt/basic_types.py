import numpy as np
from typing import Union, NamedTuple


class Integer:
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


class Float:
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


Bound = Union[Integer, Float]


class Hyperparameter(NamedTuple):
    name: str
    range: Bound

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.range.type,
            "low": self.range.low,
            "high": self.range.high
        }

    @staticmethod
    def from_dict(data: dict) -> "Hyperparameter":
        if data["type"] == "int":
            cls = Integer
            parser = int
        elif data["type"] == "float":
            cls = Float
            parser = float
        else:
            raise NotImplemented()

        return Hyperparameter(name=data["name"],
                range=cls(parser(data["low"]), parser(data["high"])))
