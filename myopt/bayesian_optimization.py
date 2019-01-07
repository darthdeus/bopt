import pickle
import concurrent.futures
from concurrent.futures import Future
from typing import Callable, List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from myopt.acquisition_functions import expected_improvement, AcquisitionFunction
from myopt.gaussian_process import GaussianProcess
from myopt.kernels import SquaredExp, Kernel
from myopt.plot import plot_approximation
from myopt.basic_types import Integer, Float, Bound, Hyperparameter
from myopt.hyperparameters import OptimizationResult


def assert_in_bounds(x: np.ndarray, bounds: List[Bound]) -> None:
    for i, bound in enumerate(bounds):
        assert bound.low <= x[i] <= bound.high, f"x_0 not in bounds, {bound} at {i}"



class OptimizationLoop:
    def __init__(self, kernel: Kernel, bounds: List[Hyperparameter], n_iter: int,
                 f: Any, optimize_kernel: bool, gp_noise: float,
                 acquisition_function: AcquisitionFunction) -> None:
        self.X_sample = np.array([], dtype=np.float32)
        self.y_sample = np.array([], dtype=np.float32)
        self.kernel = kernel.with_bounds([p.range for p in bounds])
        self.bounds = bounds
        self.n_iter = n_iter
        self.done_iter = 0
        self.f = f
        self.optimize_kernel = optimize_kernel
        self.gp_noise = gp_noise
        self.acquisition_function = acquisition_function

    def has_next(self) -> bool:
        return self.done_iter < self.n_iter

    def next(self):
        if len(self.X_sample) == 0:
            return default_from_bounds([b.range for b in self.bounds])

        gp = GaussianProcess(kernel=self.kernel, noise=self.gp_noise)
        gp.fit(self.X_sample, self.y_sample)

        if self.optimize_kernel:
            gp = gp.optimize_kernel()

        x_next = propose_location(self.acquisition_function, gp, self.y_sample.max(), self.bounds)

        return x_next

    def add_sample(self, x_next, y_next):
        assert type(y_next) == float, f"f(x) must return a float, got type {type(y_next)}, value: {y_next}"

        if len(self.X_sample) == 0 and len(self.y_sample) == 0:
            self.X_sample = np.array([x_next])
            self.y_sample = np.array([y_next])
        else:
            self.X_sample = np.vstack((self.X_sample, x_next))
            self.y_sample = np.hstack((self.y_sample, y_next))

    def result(self) -> OptimizationResult:
        X_sample = self.X_sample
        y_sample = self.y_sample
        max_y_ind = self.y_sample.argmax()

        return OptimizationResult(
            X_sample,
            y_sample,
            best_x=X_sample[max_y_ind],
            best_y=y_sample[max_y_ind],
            bounds=self.bounds,
            kernel=self.kernel.copy(),
            n_iter=self.n_iter,
            opt_fun=self.f,
        )


def default_from_bounds(bounds: List[Bound]) -> np.ndarray:
    x_0 = np.zeros(len(bounds))

    for i, bound in enumerate(bounds):
        if bound.type == "int":
            x_0[i] = np.random.randint(bound.low, bound.high)
        elif bound.type == "float":
            x_0[i] = np.random.uniform(bound.low, bound.high)
        else:
            raise RuntimeError(f"Invalid type of bound, got {type(bound)}")

    return x_0


# def bo_maximize(f: Callable[[np.array], float], bounds: List[Bound],
#                 kernel: Kernel = SquaredExp(), acquisition_function=expected_improvement,
#                 x_0: np.ndarray = None, gp_noise: float = 0,
#                 n_iter: int = 8, callback: Callable = None,
#                 optimize_kernel=True, use_tqdm=True) -> OptimizationResult:
def bo_maximize_loop(
    f: Callable[[np.array], float],
    bounds: List[Hyperparameter],
    kernel: Kernel = SquaredExp(),
    acquisition_function=expected_improvement,
    gp_noise: float = 0,
    n_iter: int = 8,
    optimize_kernel=True,
    use_tqdm=True,
) -> OptimizationResult:
    loop = OptimizationLoop(kernel, bounds, n_iter, f, optimize_kernel, gp_noise, acquisition_function)

    from tqdm import tqdm
    for i in tqdm(range(n_iter)):
        x_next = loop.next()
        y_next = f(x_next)

        loop.add_sample(x_next, y_next)

    return loop.result()


# def bo_maximize_parallel(f: Callable[[np.array], Future], bounds: List[Bound],
#                          kernel: Kernel = SquaredExp(), acquisition_function=expected_improvement,
#                          x_0: np.ndarray = None, gp_noise: float = 0,
#                          n_iter: int = 8, callback: Callable = None,
#                          optimize_kernel=True, n_parallel=3) -> OptimizationResult:
#     if x_0 is None:
#         x_0 = default_from_bounds(bounds)
#     else:
#         for i, bound in enumerate(bounds):
#             assert bound.low <= x_0[i] <= bound.high, f"x_0 not in bounds, {bound} at {i}"
#
#     kernel = kernel.with_bounds(bounds)
#
#     y_0 = f(x_0).result()
#
#     assert type(y_0) == float, "f(x) must return a float"
#
#     X_sample = np.array([x_0])
#     y_sample = np.array([y_0])
#
#     from tqdm import tqdm
#     for iter in tqdm(range((n_iter - 1) // n_parallel + 1)):
#         gp = GaussianProcess(kernel=kernel, noise=gp_noise)
#
#         multiple_x_next: List[np.ndarray] = \
#             propose_multiple_locations(acquisition_function, gp, X_sample, y_sample, bounds, n_parallel,
#                                        optimize_kernel)
#
#         futures = [f(x_next) for x_next in multiple_x_next]
#
#         completed, timeouted = concurrent.futures.wait(futures)
#
#         # We don't timeout, so this should be obvious
#         assert len(timeouted) == 0
#
#         multiple_y_next = [future.result() for future in futures]
#
#         if callback is not None:
#             callback(iter, acquisition_function, gp, X_sample, y_sample, multiple_x_next, multiple_y_next)
#
#         for x_next, y_next in zip(multiple_x_next, multiple_y_next):
#             X_sample = np.vstack((X_sample, x_next))
#             y_sample = np.hstack((y_sample, y_next))
#
#     max_y_ind = y_sample.argmax()
#     print("max_x", X_sample[max_y_ind], "max max", y_sample.max())
#
#     return OptimizationResult(X_sample=X_sample,
#                               y_sample=y_sample,
#                               opt_fun=f,
#                               n_iter=n_iter,
#                               best_x=X_sample[y_sample.argmax()],
#                               best_y=y_sample.max(),
#                               bounds=bounds,
#                               kernel=kernel.copy())
#


def bo_maximize(f: Callable[[np.array], float], params: List[Hyperparameter],
                kernel: Kernel = SquaredExp(), acquisition_function=expected_improvement,
                x_0: np.ndarray = None, gp_noise: float = 0,
                n_iter: int = 8, callback: Callable = None,
                optimize_kernel=True, use_tqdm=True) -> OptimizationResult:

    bounds = [p.range for p in params]

    if x_0 is None:
        x_0 = default_from_bounds(bounds)
    else:
        for i, bound in enumerate(bounds):
            assert bound.low <= x_0[i] <= bound.high, f"x_0 not in bounds, {bound} at {i}"

    kernel = kernel.with_bounds(bounds)

    y_0 = f(x_0)

    # TODO: handle numpy rank-0 tensors
    assert type(y_0) == float, f"f(x) must return a float, got type {type(y_0)}, value: {y_0}"

    X_sample = np.array([x_0])
    y_sample = np.array([y_0])

    iter_target = range(n_iter - 1)
    if use_tqdm:
        from tqdm import tqdm
        iter_target = tqdm(iter_target)

    for iter in iter_target:
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
    # print("max_x", X_sample[max_y_ind], "max max", y_sample.max())

    return OptimizationResult(X_sample,
                              y_sample,
                              best_x=X_sample[y_sample.argmax()],
                              best_y=y_sample.max(),
                              bounds=params,
                              kernel=kernel.copy(),
                              n_iter=n_iter,
                              opt_fun=f)


def propose_multiple_locations(acquisition: AcquisitionFunction, gp: GaussianProcess,
                               X_sample: np.ndarray, y_sample: np.ndarray,
                               bounds: List[Bound], n_locations: int, n_restarts: int = 25,
                               optimize_kernel: bool = True) -> List[np.ndarray]:
    result = []

    for _ in range(n_locations):
        gp.fit(X_sample, y_sample)
        if optimize_kernel:
            gp = gp.optimize_kernel()

        y_max = y_sample.max()

        x_next = propose_location(acquisition, gp, y_max, bounds, n_restarts)
        y_next, _ = gp.posterior(np.array([x_next])).mu_std()

        result.append(x_next)

        X_sample = np.vstack((X_sample, x_next))
        y_sample = np.hstack((y_sample, y_next))

    return result


def propose_location(acquisition: AcquisitionFunction, gp: GaussianProcess, y_max: float,
                     bounds: List[Bound], n_restarts: int = 25) -> np.ndarray:
    def min_obj(X):
        return -acquisition(gp, X.reshape(1, -1), y_max)

    scipy_bounds = [(bound.low, bound.high) for bound in bounds]

    starting_points = []
    for _ in range(n_restarts):
        starting_points.append(np.array([bound.sample() for bound in bounds]))

    min_val = 1
    min_x = None

    RUN_PARALLEL = False

    if RUN_PARALLEL:
        pass
        # results = Parallel(n_jobs=8)(
        #     delayed(minimize)(
        #         min_obj,
        #         x0=x0, bounds=scipy_bounds, method="L-BFGS-B")
        #     for x0 in starting_points)
        #
        # for res in results:
        #     if res.fun < min_val:
        #         min_val = res.fun[0]
        #         min_x = res.x
        #
        # return min_x

    else:
        for x0 in starting_points:
            res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x


def bo_plot_exploration(f: Callable[[np.ndarray], float],
                        bounds: List[Hyperparameter],
                        X_true: np.ndarray = None, y_true: np.ndarray = None,
                        kernel=SquaredExp(),
                        acquisition_function=expected_improvement,
                        n_iter: int = 8, plot_every: int = 1,
                        optimize_kernel=True, gp_noise: float = 0):
    num_plots = (n_iter // plot_every)

    plt.figure(figsize=(15, num_plots * 4))
    plt.subplots_adjust(hspace=0.4)

    def plot_iteration(i, acquisition_function, gp, X_sample, y_sample, x_next, y_next):
        ei_y = acquisition_function(gp, X_true, y_sample.max())
        per_row = 2

        if i % plot_every == 0:
            # Plot samples, surrogate function, noise-free objective and next sampling location
            ax1 = plt.subplot(num_plots // per_row + 1, per_row, i // plot_every + 1)

            plot_approximation(ax1, ei_y, X_true, y_true, gp, X_sample, y_sample,
                               [x_next], show_legend=i == 0)

            plt.title(f'Iteration {i+1}, {gp.kernel}')

    return bo_maximize(f, bounds, kernel, acquisition_function, gp_noise=gp_noise, n_iter=n_iter,
                       callback=plot_iteration, optimize_kernel=optimize_kernel)


# def bo_plot_exploration_parallel(f: Callable[[np.ndarray], Future],
#                                  bounds: List[Bound],
#                                  X_true: np.ndarray = None, y_true: np.ndarray = None,
#                                  kernel=SquaredExp(),
#                                  acquisition_function=expected_improvement,
#                                  n_iter: int = 8, plot_every: int = 1,
#                                  optimize_kernel=True, gp_noise: float = 0,
#                                  n_parallel: int = 3):
#     num_plots = (n_iter // plot_every)
#
#     plt.figure(figsize=(15, num_plots * 2))
#     plt.subplots_adjust(hspace=0.4)
#
#     def plot_iteration(i, acquisition_function, gp, X_sample, y_sample, multiple_x_next, multiple_y_next):
#         ei_y = acquisition_function(gp, X_true, y_sample.max())
#         per_row = 2
#
#         if i % plot_every == 0:
#             # Plot samples, surrogate function, noise-free objective and next sampling location
#             ax1 = plt.subplot(num_plots // per_row + 1, per_row, i // plot_every + 1)
#
#             plot_approximation(ax1, ei_y, X_true, y_true, gp, X_sample, y_sample,
#                                multiple_x_next, show_legend=i == 0)
#
#             plt.title(f'Iteration {i+1}, {gp.kernel}')
#
#     return bo_maximize_parallel(f, bounds, kernel, acquisition_function, gp_noise=gp_noise, n_iter=n_iter,
#                                 callback=plot_iteration, optimize_kernel=optimize_kernel, n_parallel=n_parallel)


def plot_2d_optim_result(result: OptimizationResult, resolution: float = 30,
                        figsize=(8, 7)):
    assert len(result.bounds) == 2

    b1 = result.bounds[0].range
    b2 = result.bounds[1].range

    x1 = np.linspace(b1.low, b1.high, resolution)
    x2 = np.linspace(b2.low, b2.high, resolution)

    assert len(x1) < 80, f"too large x1, len = {len(x1)}"
    assert len(x2) < 80, f"too large x1, len = {len(x2)}"

    gx, gy = np.meshgrid(x1, x2)

    X_2d = np.c_[gx.ravel(), gy.ravel()]

    bounds = [p.range for p in result.bounds]
    # TODO: optimize kernel

    mu, _ = GaussianProcess(kernel=result.kernel.with_bounds(bounds)) \
        .fit(result.X_sample, result.y_sample).posterior(X_2d).mu_std()

    mu_mat = mu.reshape(gx.shape[0], gx.shape[1])
    extent = [b1.low, b1.high, b2.high, b2.low]

    plt.title(f"GP posterior {round(result.best_y,2)}, {result.kernel}")
    plt.imshow(mu_mat, extent=extent, aspect="auto")
    plt.scatter(result.X_sample[:, 0], result.X_sample[:, 1], c="k")
    plt.scatter([result.best_x[0]], [result.best_x[1]], c="r")

    print(result.X_sample[0,0], result.X_sample[0, 1], result.y_sample[0])

    return mu_mat, extent, x1, x2
