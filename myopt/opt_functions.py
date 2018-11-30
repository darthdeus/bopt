from typing import NamedTuple, Callable, List

import numpy as np
from myopt.bayesian_optimization import Float, Integer, Bound

# https://www.sfu.ca/~ssurjano/franke2d.html


class OptFunction(NamedTuple):
    name: str
    max: float
    f: Callable
    parallel_f: Callable
    bounds: List[Bound]

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def beale(executor):
    """
    https://en.wikipedia.org/wiki/File:Beale%27s_function.pdf
    Max: 0
    """
    def f(x):
        y = x[1]
        x = x[0]

        val = (-1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        return -val.item()

    def parallel_f(x):
        return executor.submit(f, x)

    rng = 4.5
    bounds = [Float(-rng, rng), Float(-rng, rng)]

    return OptFunction("Beale", 0.0, f, None, bounds)


def easom(executor):
    """
    https://en.wikipedia.org/wiki/File:Easom_function.pdf
    Max: 1
    """
    def f(x):
        y = x[1]
        x = x[0]

        val = - np.cos(x) * np.cos(y) * np.exp(- ((x - np.pi)**2 + (y - np.pi)**2))
        return -val.item()

    def parallel_f(x):
        return executor.submit(f, x)

    rng = 100
    bounds = [Float(-rng, rng), Float(-rng, rng)]

    return OptFunction("Easom", 1.0, f, None, bounds)


def eggholder(executor):
    """
    https://en.wikipedia.org/wiki/File:Eggholder_function.pdf
    Max: 959.6407
    """
    def f(x):
        y = x[1]
        x = x[0]

        val = -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))
        return -val.item()

    def parallel_f(x):
        return executor.submit(f, x)

    rng = 512
    bounds = [Float(-rng, rng), Float(-rng, rng)]

    return OptFunction("Eggholder", 959.6407, f, None, bounds)


def mccormick(executor):
    """
    https://en.wikipedia.org/wiki/File:McCormick_function.pdf
    Max: 19.2085
    """
    def f(x):
        y = x[1]
        x = x[0]

        val = np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1
        return -val.item()

    def parallel_f(x):
        return executor.submit(f, x)

    bounds = [Float(-1.5, 4), Float(-3, 4)]

    return OptFunction("McCormick", 19.2085, f, None, bounds)


def get_opt_test_functions(executor):
    test_functions = [beale, easom, eggholder, mccormick]

    return [f(executor) for f in test_functions]

