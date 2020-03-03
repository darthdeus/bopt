import abc
import math
import logging
from typing import Union, NamedTuple, List, Tuple

import numpy as np


LOGSCALE_BASE = 10.0


ParamTypes = Union[float, int, str]


class OptimizationFailed(Exception):
    pass


# TODO: fix naming convention & typnig errors
class Bound(abc.ABC):
    low: float
    high: float
    buckets: int
    type: str

    @abc.abstractmethod
    def __init__(self, low: float, high: float, buckets: int) -> None:
        pass

    @abc.abstractmethod
    def map(self, value) -> ParamTypes:
        pass

    @abc.abstractmethod
    def inverse_map(self, value) -> ParamTypes:
        pass

    @abc.abstractmethod
    def is_discrete(self) -> bool:
        pass

    @abc.abstractmethod
    def is_logscale(self) -> bool:
        pass

    # TODO: is this before or after transform?
    @abc.abstractmethod
    def sample(self) -> ParamTypes:
        pass

    @abc.abstractmethod
    def validate(self, value: ParamTypes) -> bool:
        pass

    @abc.abstractmethod
    def should_transform(self) -> bool:
        pass

    @abc.abstractmethod
    def parse(self, value: str) -> ParamTypes:
        return value

    @abc.abstractmethod
    def scipy_bound_tuple(self) -> Tuple[float, float]:
        pass

    @abc.abstractmethod
    def compare_values(self, a: ParamTypes, b: ParamTypes) -> bool:
        pass

    @abc.abstractmethod
    def maybe_round(self, value: np.ndarray) -> np.ndarray:
        pass

    def grid(self, resolution: int) -> np.ndarray:
        if self.is_logscale():
            grid = np.linspace(math.log10(self.low), math.log10(self.high), num=resolution)
        else:
            grid = np.linspace(self.low, self.high, num=resolution)

        return grid

    def round_buckets(self, value: np.ndarray) -> np.ndarray:
        assert self.buckets > 0, "Bucket rounding can be done only when num buckets > 0"
        _, bins = np.histogram([self.low, self.high], bins=self.buckets)
        bins[-1] += 1e-6

        idx = np.digitize(value, bins)
        return bins[idx]


class Integer(Bound):
    def __init__(self, low: int, high: int, buckets: int):
        self.low = low
        self.high = high
        self.buckets = buckets
        self.type = "int"

    def sample(self) -> float:
        return np.random.randint(self.low, self.high + 1)

    def map(self, value) -> ParamTypes:
        return value

    def inverse_map(self, value) -> ParamTypes:
        return round(value)

    def is_discrete(self) -> bool:
        return True

    def is_logscale(self) -> bool:
        return False

    def should_transform(self) -> bool:
        return False

    def validate(self, value: ParamTypes) -> bool:
        assert isinstance(value, int), "expected int, got {}".format(value)
        # TODO: ma byt <= ... <, ale z nejakeho duvodu to failuje
        return self.low <= value <= self.high

    def __repr__(self) -> str:
        return f"Int({self.low}, {self.high})"

    def parse(self, value: str) -> ParamTypes:
        return int(value)

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        # TODO: pryc s -1?
        # return (self.low, (self.high - 1))
        return self.low, self.high

    def compare_values(self, a: ParamTypes, b: ParamTypes) -> bool:
        assert isinstance(a, int)
        assert isinstance(b, int)

        return a == b

    def maybe_round(self, value: np.ndarray) -> np.ndarray:
        if self.buckets > 0:
            value = super().round_buckets(value)

        # logging.warn("TODO: why is this ever a numpy thing and not an int?")
        return np.floor(value).astype(np.int32)
        # floored = np.floor(value)
        # if isinstance(value, float):
        #     return int(floored)
        # else:
        #     return floored


class Float(Bound):
    def __init__(self, low: float, high: float, buckets: int):
        self.low = low
        self.high = high
        self.buckets = buckets
        self.type = "float"

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

    def map(self, value) -> ParamTypes:
        return value

    def inverse_map(self, value) -> ParamTypes:
        return value

    def is_discrete(self) -> bool:
        return False

    def is_logscale(self) -> bool:
        return False

    def should_transform(self) -> bool:
        return False

    def validate(self, value: ParamTypes) -> bool:
        assert isinstance(value, float)
        return self.low <= value <= self.high

    def __repr__(self) -> str:
        return f"Float({self.low}, {self.high})"

    def parse(self, value: str) -> ParamTypes:
        return float(value)

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        return self.low, self.high

    def compare_values(self, a: ParamTypes, b: ParamTypes) -> bool:
        assert isinstance(a, float)
        assert isinstance(b, float)

        # diff = abs(a - b)
        # TODO: logscale will need this adjusted
        # We set the threshold at 1% of the range
        # threshold = (self.high - self.low) * 0.01
        # return diff < threshold
        return False

    def maybe_round(self, value: np.ndarray) -> np.ndarray:
        if self.buckets == -1:
            return value
        else:
            return super().round_buckets(value)


class LogscaleInt(Bound):
    def __init__(self, low: float, high: float, buckets: int):
        self.low = low
        self.high = high
        self.buckets = buckets
        self.type = "logscale_int"

    def sample(self) -> float:
        return int(LOGSCALE_BASE ** np.random.uniform(np.log10(self.low), np.log10(self.high)))

    def is_discrete(self) -> bool:
        # TODO: fuj, prejmenovat na neco jako should_round_before_map? :D
        return False

    def is_logscale(self) -> bool:
        return True

    def should_transform(self) -> bool:
        return True

    def validate(self, value: ParamTypes) -> bool:
        assert isinstance(value, int), "value {} is not int".format(value)
        # TODO: podobne jako u Integer tady ma byt asi < ?
        return self.low <= value <= self.high

    def __repr__(self) -> str:
        return f"LogscaleInt({self.low}, {self.high})"

    def map(self, value) -> float:
        return np.log10(value)

    def inverse_map(self, value) -> int:
        return round(LOGSCALE_BASE ** value)

    def parse(self, value: str) -> ParamTypes:
        return int(value)

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        return np.log10(self.low), np.log10(self.high)

    def compare_values(self, a: ParamTypes, b: ParamTypes) -> bool:
        assert isinstance(a, int)
        assert isinstance(b, int)

        return a == b

    # def maybe_round(self, value: np.ndarray) -> np.ndarray:
    #     # TODO: round in logspace
    #     if self.buckets == -1:
    #         return np.floor(value)
    #     else:
    #         _, bins = np.histogram([self.low, self.high], bins=self.buckets)
    #         bins[-1] += 1e-6
    #
    #         idx = np.digitize(value, bins)
    #         return bins[idx]
    def maybe_round(self, value: np.ndarray) -> np.ndarray:
        if self.buckets > 0:
            value = super().round_buckets(value)

        return np.floor(value).astype(np.int32)


class LogscaleFloat(Bound):
    def __init__(self, low: float, high: float, buckets: int):
        self.low = low
        self.high = high
        self.buckets = buckets
        self.type = "logscale_float"

    def sample(self) -> float:
        return LOGSCALE_BASE ** np.random.uniform(np.log10(self.low), np.log10(self.high))

    def is_discrete(self) -> bool:
        return False

    def is_logscale(self) -> bool:
        return True

    def should_transform(self) -> bool:
        return True

    def validate(self, value: ParamTypes) -> bool:
        assert isinstance(value, float)
        return self.low <= value <= self.high

    def __repr__(self) -> str:
        return f"LogscaleFloat({self.low}, {self.high})"

    def map(self, value) -> float:
        return np.log10(value)

    def inverse_map(self, value) -> float:
        return LOGSCALE_BASE ** value

    def parse(self, value: str) -> ParamTypes:
        return float(value)

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        return np.log10(self.low), np.log10(self.high)

    def compare_values(self, a: ParamTypes, b: ParamTypes) -> bool:
        assert isinstance(a, float)
        assert isinstance(b, float)

        diff = abs(a - b)
        # TODO: logscale will need this adjusted
        # We set the threshold at 1% of the range
        threshold = (np.log10(self.high) - np.log10(self.low)) * 0.01
        return diff < threshold

    def maybe_round(self, value: np.ndarray) -> np.ndarray:
        if self.buckets == -1:
            return value
        else:
            # TODO: logscale rounding
            return super().round_buckets(value)


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

    def is_logscale(self) -> bool:
        return False

    def should_transform(self) -> bool:
        return True

    def map(self, value) -> int:
        return self.values.index(value)

    def inverse_map(self, value) -> str:
        return self.values[int(value)]

    def __repr__(self) -> str:
        return f"Discrete({self.values}, {self.low}, {self.high})"

    def parse(self, value: str) -> ParamTypes:
        return value

    def scipy_bound_tuple(self) -> Tuple[float, float]:
        return (self.low, (self.high - 1))

    def compare_values(self, a: ParamTypes, b: ParamTypes) -> bool:
        assert isinstance(a, str)
        assert isinstance(b, str)

        return a == b

    def maybe_round(self, value: np.ndarray) -> np.ndarray:
        return np.floor(value)


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
                "high": self.range.high,
                "buckets": self.range.buckets,
            }

    def validate(self, value) -> bool:
        return self.range.validate(value)

    def maybe_round(self, value: np.ndarray) -> np.ndarray:
        return self.range.maybe_round(value)

    @staticmethod
    def from_dict(name, data: dict) -> "Hyperparameter":
        if data["type"] == "discrete":
            return Hyperparameter(name=name, range=Discrete(data["values"]))
        else:
            mapping = {
                "int": [Integer, int],
                "logscale_int": [LogscaleInt, int],
                "float": [Float, float],
                "logscale_float": [LogscaleFloat, float]
            }

            if data["type"] in mapping:
                cls, parser = mapping[data["type"]]
                range = cls(parser(data["low"]), parser(data["high"]), int(data["buckets"]))

                return Hyperparameter(name=name, range=range)
            else:
                raise NotImplementedError()
        # elif data["type"] == "int":
        #     return Hyperparameter(name=name,
        #             range=Integer(int(data["low"]), int(data["high"])))
        # elif data["type"] == "logscale_float":
        #     return Hyperparameter(name=name,
        #             range=LogscaleFloat(float(data["low"]), float(data["high"])))
        # elif data["type"] == "logscale_int":
        #     return Hyperparameter(name=name,
        #             range=LogscaleInt(int(data["low"]), int(data["high"])))
        # elif data["type"] == "float":
        #     return Hyperparameter(name=name,
        #             range=Float(float(data["low"]), float(data["high"])))
