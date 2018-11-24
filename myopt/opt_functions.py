import numpy as np
from bayesian_optimization import Float, Integer


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

    return f, parallel_f, bounds


def easom(executor):
    """
    https://en.wikipedia.org/wiki/File:Easom_function.pdf
    Max: 0
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

    return f, parallel_f, bounds


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

    return f, parallel_f, bounds


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

    return f, parallel_f, bounds


test_functions = [beale, eggholder, mccormick]