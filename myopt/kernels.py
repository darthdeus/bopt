import abc
from typing import Optional

import numpy as np


FAST_KERNEL = True


class Kernel(abc.ABC):
    round_indexes: Optional[np.ndarray]

    def __init__(self):
        self.round_indexes = None

    def __call__(self, x_init: np.ndarray, y_init: np.ndarray = None) -> np.ndarray:
        if y_init is None:
            y_init = x_init

        if FAST_KERNEL:
            x = x_init
            y = y_init

            if x.ndim > 2 or y.ndim > 2:
                raise RuntimeError(f"Invalid input, can only handle rank 1 or 2 tensors, got {x.ndim}, {y.ndim}.")

            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            x = x.copy()
            y = y.copy()

            if self.round_indexes is not None and len(self.round_indexes) > 0:
                x[:, self.round_indexes] = np.round(x[:, self.round_indexes])
                y[:, self.round_indexes] = np.round(y[:, self.round_indexes])

            x = x.reshape(x.shape[0], 1, x.shape[1])
            y = y.reshape(1, y.shape[0], y.shape[1])

            output = self.kernel(x, y) + 1e-12 * np.eye(x.shape[0], y.shape[1])
        else:
            #
            # # <a,1,h>
            # # <1,b,h>
            x = x_init
            y = y_init

            output = np.zeros((len(x), len(y)), dtype=np.float64)

            for i in range(len(x)):
                for j in range(len(y)):
                    output[i, j] = self.kernel(x[i], y[j])

                    if i == j:
                        output[i, j] += 1e-12

        assert output.shape == (x_init.shape[0], y_init.shape[0])
        return output

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
    def set_params(self, theta: list) -> None:
        pass

    def with_params(self, theta: list) -> "Kernel":
        kernel = self.copy()
        kernel.set_params(theta)
        return kernel

    def with_round_indexes(self, indexes: np.ndarray) -> "Kernel":
        kernel = self.copy()
        kernel.round_indexes = indexes
        return kernel

    def with_bounds(self, bounds) -> "Kernel":
        kernel = self.copy()
        kernel.round_indexes = np.array([i for i, bound in enumerate(bounds) if bound.type == "int"])
        return kernel

    @abc.abstractmethod
    def copy(self) -> "Kernel":
        pass


class SquaredExp(Kernel):
    def __init__(self, l: float = 1, sigma: float = 1) -> None:
        super().__init__()
        self.l = l
        self.sigma = sigma

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if FAST_KERNEL:
            sqnorm = ((x - y) ** 2).sum(axis=2)

            assert x.shape[1] == 1
            assert y.shape[0] == 1
            assert sqnorm.shape[0] == x.shape[0]
            assert sqnorm.shape[1] == y.shape[1]

            return self.sigma ** 2 * np.exp(- (1 / (2*self.l**2)) * sqnorm)

        else:
            return self.sigma ** 2 * np.exp(- (1 / (2 * self.l ** 2)) * (x * x + y * y - 2 * x * y))

    def default_params(self) -> np.ndarray:
        return np.array([1, 1])

    def param_bounds(self) -> list:
        return [(1e-5, None), (1e-5, None)]

    def set_params(self, theta) -> None:
        self.l = theta[0]
        self.sigma = theta[1]

    def copy(self) -> "Kernel":
        copy = SquaredExp(l=self.l, sigma=self.sigma)
        copy.round_indexes = self.round_indexes
        return copy

    def __repr__(self):
        return f"SquaredExp(l={round(self.l, 5)}, sigma={round(self.sigma, 2)})"


class Matern(Kernel):
    def __init__(self, sigma: float = 1, ro: float = 1):
        super().__init__()
        self.sigma = sigma
        self.ro = ro

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sqnorm = ((x - y) ** 2).sum(axis=2)

        assert x.shape[1] == 1
        assert y.shape[0] == 1
        assert sqnorm.shape[0] == x.shape[0]
        assert sqnorm.shape[1] == y.shape[1]

        d = np.sqrt(sqnorm)

        # zavorka = (np.sqrt(2 * self.v) * sqnorm / ro)
        # e1 = sigma**2 * (2 ** (1 - self.v))/(gamma(self.v))
        # e2 = zavorka ** self.v
        # e3 = kv(self.v, zavorka)
        #
        # return e1 * e2 * e3
        sigma = self.sigma
        ro = self.ro

        return sigma**2 * (1 + (np.sqrt(5) * d) / ro + (5*d**2)/(3*ro**2)) * np.exp(-(np.sqrt(5)*d)/(ro))

    def default_params(self) -> np.ndarray:
        return np.array([1, 1])

    def param_bounds(self) -> list:
        return [(1e-5, None), (1e-5, None)]

    def set_params(self, theta: list) -> None:
        self.sigma = theta[0]
        self.ro = theta[1]

    def copy(self) -> "Kernel":
        return Matern(self.sigma, self.ro)

    def __repr__(self):
        return f"Matern(sigma={self.sigma}, ro={self.ro})"


class RationalQuadratic(Kernel):
    def __init__(self, sigma: float = 1, l: float = 1, alpha: float = 1) -> None:
        super().__init__()
        self.sigma = sigma
        self.l = l
        self.alpha = alpha

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sqnorm = ((x - y) ** 2).sum(axis=2)
        return self.sigma**2 * (1 + sqnorm / (2 * self.alpha * self.l**2)) **(-self.alpha)

    def default_params(self) -> np.ndarray:
        return np.array([1])

    def param_bounds(self) -> list:
        return [(1e-5, None)]

    def set_params(self, theta: list) -> None:
        self.l = theta[0]

    def copy(self) -> "Kernel":
        return RationalQuadratic(sigma=self.sigma, l=self.l, alpha=self.alpha)

    def __repr__(self):
        return f"RationalQuadratic(sigma={self.sigma}, l={self.l}, alpha={self.alpha})"


class Linear(Kernel):
    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        yy = y.T

        rows = []

        for i in range(x.shape[0]):
            rows.append(x[i] @ yy)

        return np.array(rows)

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
