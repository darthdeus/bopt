from typing import NamedTuple, Callable, List

import numpy as np
from myopt.bayesian_optimization import Float, Integer, Bound

# https://www.sfu.ca/~ssurjano/franke2d.html


class OptFunction(NamedTuple):
    name: str
    max: float
    f: Callable
    bounds: List[Bound]

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)



def beale(x: np.ndarray) -> float:
    """
    https://en.wikipedia.org/wiki/File:Beale%27s_function.pdf
    Max: 0
    """
    y = x[1]
    x = x[0]

    val = (-1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    return -val.item()

beale.name = "Beale"
beale.bounds = [Float(-4.5, 4.5), Float(-4.5, 4.5)]


def easom(x: np.ndarray) -> float:
    """
    https://en.wikipedia.org/wiki/File:Easom_function.pdf
    Max: 1
    """
    y = x[1]
    x = x[0]

    val = - np.cos(x) * np.cos(y) * np.exp(- ((x - np.pi)**2 + (y - np.pi)**2))
    return -val.item()

easom.name = "Easom"
easom.bounds = [Float(-100, 100), Float(-100, 100)]


def eggholder(x: np.ndarray) -> float:
    """
    https://en.wikipedia.org/wiki/File:Eggholder_function.pdf
    Max: 959.6407
    """
    y = x[1]
    x = x[0]

    val = -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return -val.item()

eggholder.name = "Eggholder"
eggholder.bounds = [Float(-512, 512), Float(-512, 512)]


def mccormick(x: np.ndarray) -> float:
    """
    https://en.wikipedia.org/wiki/File:McCormick_function.pdf
    Max: 19.2085
    """
    y = x[1]
    x = x[0]

    val = np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1
    return -val.item()

mccormick.name = "McCormick"
mccormick.bounds = [Float(-1.5, 4), Float(-3, 4)]


def get_fun_by_name(name: str):
    funs = get_opt_test_functions()

    return [fun for fun in funs if fun.name == name][0]


def get_opt_test_functions():
    return [beale, easom, eggholder, mccormick]

