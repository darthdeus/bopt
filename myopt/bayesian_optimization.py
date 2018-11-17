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


class OptimizationResult:
    X_sample: np.ndarray
    y_sample: np.ndarray
    best_x: np.ndarray
    best_y: float
    bounds: List[Bound]
    kernel: Kernel

    def __init__(self, X_sample: np.ndarray, y_sample: np.ndarray, best_x: np.ndarray, best_y: float,
                 bounds: List[Bound], kernel: Kernel) -> None:
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.best_x = best_x
        self.best_y = best_y
        self.bounds = bounds
        self.kernel = kernel

    def __repr__(self) -> str:
        return f"OptimizationResult(best_x={self.best_x}, best_y={self.best_y})"


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
        y_sample = np.hstack((y_sample, y_next))

    max_y_ind = y_sample.argmax()
    print("max_x", X_sample[max_y_ind], "max max", y_sample.max())

    return OptimizationResult(X_sample,
                              y_sample,
                              best_x=X_sample[y_sample.argmax()],
                              best_y=y_sample.max(),
                              bounds=bounds,
                              kernel=kernel.copy())


def propose_location(acquisition: Callable, gp: GaussianProcess, y_max: float, bounds: List[Bound], n_restarts: int=25):
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        val = -acquisition(gp, X.reshape(1, -1), y_max)
        return val

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


def plot_2d_optim_result(result: OptimizationResult, resolution: float = 0.1):
    assert len(result.bounds) == 2

    b1 = result.bounds[0]
    b2 = result.bounds[1]

    x1 = np.arange(b1.low, b1.high, resolution)
    x2 = np.arange(b2.low, b2.high, resolution)

    gx, gy = np.meshgrid(x1, x2)

    X_2d = np.c_[gx.ravel(), gy.ravel()]

    mu, _ = GaussianProcess(kernel=result.kernel).fit(result.X_sample, result.y_sample).posterior(X_2d).mu_std()

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title("GP posterior")
    plt.imshow(mu.reshape(gx.shape[0], gy.shape[0]),
               extent=[b1.low, b1.high, b2.low, b2.high])
    plt.scatter(result.X_sample[:,0], result.X_sample[:,1], c="k")
    plt.scatter([result.best_x[0]], [result.best_x[1]], c="r")