import abc
from typing import List

import numpy as np


def sqexp(x, y, l=1):
    return np.exp(-.5 * l * (x * x + y * y - 2 * x * y))


class Kernel(abc.ABC):
    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if y is None:
            y = x

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        return self.kernel(*np.meshgrid(y, x)) + 1e-12 * np.eye(x.shape[0], y.shape[0])

    @abc.abstractmethod
    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def default_params(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def param_bounds(self) -> list:
        pass

    @abc.abstractmethod
    def with_params(self, theta: list) -> "Kernel":
        pass


class SquaredExp(Kernel):
    def __init__(self, l=1) -> None:
        super().__init__()
        self.l = l

    def with_params(self, theta) -> "Kernel":
        return SquaredExp(theta[0])

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-.5 * self.l * (x * x + y * y - 2 * x * y))

    def default_params(self) -> np.ndarray:
        return np.ndarray([1])

    def param_bounds(self) -> list:
        return [(1e-5, None)]


def k(x, y=None, kernel=sqexp):
    if y is None:
        y = x

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return kernel(*np.meshgrid(y, x)) + 1e-12 * np.eye(x.shape[0], y.shape[0])