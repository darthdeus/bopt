import abc
import numpy as np


FAST_KERNEL = False


class Kernel(abc.ABC):
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

    # def default_params(self) -> np.ndarray:
    #     return np.array([1])
    #
    # def param_bounds(self) -> list:
    #     return [(1e-3, None)]
    #
    # def with_params(self, theta) -> "Kernel":
    #     return SquaredExp(theta[0])

    def default_params(self) -> np.ndarray:
        return np.array([1, 1])

    def param_bounds(self) -> list:
        return [(1e-5, None), (1e-5, None)]

    def with_params(self, theta) -> "Kernel":
        return SquaredExp(theta[0], theta[1])

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
        sqnorm = ((x - y) ** 2).sum(axis=2)
        return self.sigma**2 * (1 + sqnorm / (2 * self.alpha * self.l**2)) **(-self.alpha)

    def default_params(self) -> np.ndarray:
        return np.array([1])

    def param_bounds(self) -> list:
        return [(1e-5, None)]

    def with_params(self, theta: list) -> "Kernel":
        return RationalQuadratic(l=theta[0])

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
