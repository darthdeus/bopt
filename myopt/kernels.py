import abc
from typing import Union

import numpy as np


# TODO: handle floats for extra laziness?
Arrayable = Union[list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


def sqexp(x: np.ndarray, y: np.ndarray, l: float = 1):
    return np.exp(-.5 * l * (x * x + y * y - 2 * x * y))


class Kernel(abc.ABC):
    def __call__(self, x: Arrayable, y: Arrayable = None) -> np.ndarray:
        if y is None:
            y = x

        x = ensure_array(x)
        y = ensure_array(y)

        x = x.reshape(len(x), -1)
        y = y.reshape(len(y), -1)

        output = np.zeros((len(x), len(y)), dtype=np.float64)

        for i in range(len(x)):
            for j in range(len(y)):
                output[i, j] = self.kernel(x[i], y[j])

                if i == j:
                    output[i, j] += 1e-12

        return output

        # return self.kernel(x, y)
        # return self.kernel(*np.meshgrid(y, x)) + 1e-12 * np.eye(x.shape[0], y.shape[0])

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


class Linear(Kernel):
    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x @ y

    def default_params(self) -> np.ndarray:
        return np.array([])

    def param_bounds(self) -> list:
        return []

    def with_params(self, theta: list) -> "Kernel":
        return self


class SquaredExp(Kernel):
    def __init__(self, l: float = 1) -> None:
        super().__init__()
        self.l = l

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-.5 * self.l * (x * x + y * y - 2 * x * y))

    def default_params(self) -> np.ndarray:
        return np.ndarray([1])

    def param_bounds(self) -> list:
        return [(1e-5, None)]

    def with_params(self, theta) -> "Kernel":
        return SquaredExp(theta[0])

    def __repr__(self):
        return f"SquaredExp(l={self.l})"


def k(x, y=None, kernel=sqexp):
    if y is None:
        y = x

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return kernel(*np.meshgrid(y, x)) + 1e-12 * np.eye(x.shape[0], y.shape[0])