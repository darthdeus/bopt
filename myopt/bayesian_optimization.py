import functools
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .acquisition_functions import expected_improvement
from .gaussian_process import GaussianProcess
from .kernels import SquaredExp, Kernel
from .plot import plot_approximation, plot_convergence


class Integer:
    low: int
    high: int

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.type = "int"

    def sample(self) -> float:
        return np.random.randint(self.low, self.high)


class Float:
    low: float
    high: float

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.type = "float"

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)


Bound = Union[Integer, Float]


def bo_minimize(f: Callable[[np.array], float], bounds: List[Bound],
                kernel: Kernel = SquaredExp(), acquisition_function=expected_improvement,
                x_0: np.ndarray = None, gp_noise: float = 0,
                n_iter: int = 8, callback: Callable = None,
                optimize_kernel=True):
    if x_0 is None:
        x_0 = np.zeros(len(bounds))

        for i, bound in enumerate(bounds):
            if bound.type == "int":
                x_0[i] = np.random.randint(bound.low, bound.high)
            elif bound.type == "float":
                x_0[i] = np.random.uniform(bound.low, bound.high)
            else:
                raise RuntimeError(f"Invalid type of bound, got {type(bound)}")
    else:
        for i, bound in enumerate(bounds):
            assert bound.low <= x_0[i] <= bound.high, f"x_0 not in bounds, {bound} at {i}"

    kernel.round_indexes = np.array([i for i, bound in enumerate(bounds) if bound.type == "int"])

    y_0 = f(x_0)

    X_sample = np.array([x_0])
    y_sample = np.array([y_0])

    for iter in range(n_iter - 1):
        gp = GaussianProcess(kernel=kernel, noise=gp_noise).fit(X_sample, y_sample)
        if optimize_kernel:
            gp = gp.optimize_kernel()

        x_next = propose_location(acquisition_function, gp, y_sample.max(), bounds)

        y_next = f(x_next)

        if callback is not None:
            callback(iter, acquisition_function, gp, X_sample, y_sample, x_next, y_next)

        X_sample = np.vstack((X_sample, x_next))
        y_sample = np.vstack((y_sample, y_next))

    max_y_ind = y_sample.argmax()
    print("max_x", X_sample[max_y_ind], "max max", y_sample.max())

    return X_sample[y_sample.argmax()]


def propose_location(acquisition: Callable, gp: GaussianProcess, y_max: float, bounds: List[Bound], n_restarts: int=25):
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(gp, X, y_max)

    starting_points = []
    for _ in range(n_restarts):
        starting_points.append(np.array([bound.sample() for bound in bounds]))

    scipy_bounds = [(bound.low, bound.high) for bound in bounds]

    # Find the best optimum by starting from n_restart different random points.
    for x0 in starting_points:
        res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x


def bo_plot_exploration(f: Callable[[np.ndarray], np.ndarray],
                        bounds: List[Bound],
                        X_init: np.ndarray, y_init: np.ndarray,
                        X_true: np.ndarray = None, y_true: np.ndarray = None,
                        kernel=SquaredExp(),
                        acquisition_function=expected_improvement,
                        n_iter: int = 8, plot_every: int = 1,
                        optimize_kernel=True, gp_noise: float = 0):
    num_plots = (n_iter // plot_every)

    plt.figure(figsize=(15, num_plots * 2))
    plt.subplots_adjust(hspace=0.4)

    def plot_iteration(i, acquisition_function, gp, X_sample, y_sample, x_next, y_next):
        ei_y = acquisition_function(gp, X_true, y_sample.max())
        per_row = 2

        if i % plot_every == 0:
            # Plot samples, surrogate function, noise-free objective and next sampling location
            ax1 = plt.subplot(num_plots // per_row + 1, per_row, i // plot_every + 1)

            plot_approximation(ax1, ei_y, gp.kernel, X_true, y_true, gp, X_sample, y_sample,
                               x_next, show_legend=i == 0)

            plt.title(f'Iteration {i+1}, {gp.kernel}')

    return bo_minimize(f, bounds, kernel, acquisition_function, gp_noise=gp_noise, n_iter=n_iter,
                       callback=plot_iteration, optimize_kernel=optimize_kernel)

    # plot_convergence(X_sample, y_sample)


#
#
# def bo_plot_exploration(f: Callable[[np.ndarray], np.ndarray],
#                         bounds: List[Bound],
#                         X_init: np.ndarray, y_init: np.ndarray,
#                         X_true: np.ndarray = None, y_true: np.ndarray = None,
#                         kernel=SquaredExp(),
#                         acquisition_function=expected_improvement,
#                         n_iter: int = 8, plot_every: int = 1,
#                         optimize_kernel=True, gp_noise: float = 0):
#     num_plots = (n_iter // plot_every)
#
#     plt.figure(figsize=(15, num_plots * 2))
#     plt.subplots_adjust(hspace=0.4)
#
#     def plot_iteration(i, acquisition_function, gp, X_sample, y_sample, x_next, y_next):
#         ei_y = acquisition_function(gp, X, X_true)
#         per_row = 2
#
#         if i % plot_every == 0:
#             # Plot samples, surrogate function, noise-free objective and next sampling location
#             ax1 = plt.subplot(num_plots // per_row + 1, per_row, i // plot_every + 1)
#
#             plot_approximation(ax1, ei_y, gp.kernel, X_true, y_true, gp, X_sample, y_sample,
#                                x_next, show_legend=i == 0)
#
#             plt.title(f'Iteration {i+1}, {gp.kernel}')
#
#     bo_minimize(f, bounds, kernel, acquisition_function, gp_noise=gp_noise, n_iter=n_iter,
#                 callback=plot_iteration, optimize_kernel=optimize_kernel)
#
#     # plot_convergence(X_sample, y_sample)
