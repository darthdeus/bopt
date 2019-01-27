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
        return np.random.randint(self.low, self.high).item()

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
        return np.random.uniform(self.low, self.high).item()

    def __repr__(self) -> str:
        return f"Float({self.low}, {self.high})"


Bound = Union[Integer, Float]


class Hyperparameter(NamedTuple):
  name: str
  range: Bound

