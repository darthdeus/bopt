from typing import NamedTuple, Callable, List

import abc
import numpy as np
import sys
from bopt.basic_types import Float, Integer, Bound

# TODO: Rewrite all Bounds to Hyperparmeters

# https://www.sfu.ca/~ssurjano/franke2d.html


class OptFunction(abc.ABC):
    name: str
    max: float
    f: Callable[[np.ndarray], float]
    bounds: List[Bound]

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class Beale(OptFunction):
    name = "Beale"

    def __init__(self) -> None:
        self.bounds = [Float(-4.5, 4.5, -1), Float(-4.5, 4.5, -1)]

    def f(self, x: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/File:Beale%27s_function.pdf
        Max: 0
        """
        y = x[1]
        x = x[0]

        val = (-1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        return -val.item()


class Easom(OptFunction):
    name = "Easom"

    def __init__(self) -> None:
        self.bounds = [Float(-100, 100, -1), Float(-100, 100, -1)]

    def f(self, x: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/File:Easom_function.pdf
        Max: 1
        """
        y = x[1]
        x = x[0]

        val = - np.cos(x) * np.cos(y) * np.exp(- ((x - np.pi)**2 + (y - np.pi)**2))
        return -val.item()


class Eggholder(OptFunction):
    name = "Eggholder"

    def __init__(self) -> None:
        self.bounds = [Float(-512, 512, -1), Float(-512, 512, -1)]

    def f(self, x: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/File:Eggholder_function.pdf
        Max: 959.6407
        """
        y = x["y"]
        x = x["x"]

        e1 = -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47))))
        e2 = x * np.sin(np.sqrt(np.abs(x - (y + 47))))

        val = e1 - e2
        return -val.item()


class McCormick(OptFunction):
    name = "McCormick"

    def __init__(self) -> None:
        self.bounds = [Float(-1.5, 4, -1), Float(-3, 4, -1)]

    def f(self, x: np.ndarray) -> float:
        """
        https://en.wikipedia.org/wiki/File:McCormick_function.pdf
        Max: 19.2085
        """
        y = x[1]
        x = x[0]

        val = np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1
        return -val.item()


class XSquared(OptFunction):
    name = "XSquared"

    def __init__(self) -> None:
        self.bounds = [Float(0.1, 6, -1), Float(0.1, 6, -1)]

    def f(self, x: np.ndarray) -> float:
        """
        Max: 2
        """
        y = x[1] + 3
        x = x[0] + 3

        val = -(x*x + y*y - 2)
        return -val.item()


def get_fun_by_name(name: str):
    funs = get_opt_test_functions()
    return [fun for fun in funs if fun.name == name][0]


def get_opt_test_functions():
    # return [Beale(), Easom(), Eggholder(), McCormick()]
    return [XSquared()]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the function", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)

    args = parser.parse_args()

    for fun in get_opt_test_functions():
        if fun.name == args.name:
            result = fun.f(np.array([args.x, args.y]))
            print("RESULT={}".format(result))
            sys.exit(0)

    print("Invalid function name {}".format(args.name), file=sys.stderr)
    sys.exit(1)
