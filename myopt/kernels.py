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

        # output = np.zeros((len(x), len(y)), dtype=np.float64)
        #
        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         output[i, j] = self.kernel(x[i], y[j])
        #
        #         if i == j:
        #             output[i, j] += 1e-12
        #
        # return output

        # return self.kernel(x, y)
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

    @abc.abstractmethod
    def copy(self) -> "Kernel":
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

    def copy(self) -> "Kernel":
        return Linear()

    def __repr__(self):
        return f"Linear()"


class SquaredExp(Kernel):
    def __init__(self, l: float = 1, sigma: float = 1) -> None:
        super().__init__()
        self.l = l
        self.sigma = sigma

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.sigma ** 2 * np.exp(- (1 / 2 * self.l ** 2) * (x * x + y * y - 2 * x * y))

    def default_params(self) -> np.ndarray:
        return np.array([1])

    def param_bounds(self) -> list:
        return [(1e-5, None)]

    def with_params(self, theta) -> "Kernel":
        return SquaredExp(theta[0])

    # def default_params(self) -> np.ndarray:
    #     return np.ndarray([2])
    #
    # def param_bounds(self) -> list:
    #     return [(1e-5, None), (1e-5, None)]
    #
    # def with_params(self, theta) -> "Kernel":
    #     return SquaredExp(theta[0], theta[1])

    def copy(self) -> "Kernel":
        return SquaredExp(l=self.l, sigma=self.sigma)

    def __repr__(self):
        return f"SquaredExp(l={round(self.l, 2)}, sigma={round(self.sigma, 2)})"


class RationalQuadratic(Kernel):
    def __init__(self, sigma: float = 1, l: float = 1, alpha: float = 1) -> None:
        self.sigma = sigma
        self.l = l
        self.alpha = alpha

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.sigma**2 * (1 + (x - y)**2 / (2 * self.alpha * self.l**2)) **(-self.alpha)

    def default_params(self) -> np.ndarray:
        return np.ndarray([1])

    def param_bounds(self) -> list:
        return [(1e-5, None)]

    def with_params(self, theta: list) -> "Kernel":
        return RationalQuadratic(l=theta[0])

    def copy(self) -> "Kernel":
        return RationalQuadratic(sigma=self.sigma, l=self.l, alpha=self.alpha)

    def __repr__(self):
        return f"RationalQuadratic(sigma={self.sigma}, l={self.l}, alpha={self.alpha})"

def k(x, y=None, kernel=sqexp):
    if y is None:
        y = x

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return kernel(*np.meshgrid(y, x)) + 1e-12 * np.eye(x.shape[0], y.shape[0])
