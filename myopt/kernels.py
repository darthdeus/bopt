import numpy as np


def sqexp(x, y, l=1):
    return np.exp(-.5 * l * (x * x + y * y - 2 * x * y))


def k(x, y=None, kernel=sqexp):
    if y is None:
        y = x

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return kernel(*np.meshgrid(y, x)) + 1e-12 * np.eye(x.shape[0], y.shape[0])