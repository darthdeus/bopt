from typing import Callable, List, Union, Any, Dict, Tuple

import psutil
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from bopt.acquisition_functions import expected_improvement, AcquisitionFunction
from bopt.gaussian_process import GaussianProcess
from bopt.kernels import SquaredExp, Kernel
from bopt.plot import plot_approximation
from bopt.basic_types import Integer, Float, Bound, Hyperparameter
from bopt.hyperparameters import OptimizationResult


def assert_in_bounds(x: np.ndarray, bounds: List[Bound]) -> None:
    for i, bound in enumerate(bounds):
        assert bound.low <= x[i] <= bound.high, f"x_0 not in bounds, {bound} at {i}"


class OptimizationLoop:
    def __init__(self, params: List[Hyperparameter],
                 kernel: Kernel = SquaredExp(), optimize_kernel: bool = True,
                 gp_noise: float = 0.0,
                 acquisition_function: AcquisitionFunction = expected_improvement) -> None:
        self.X_sample = np.array([], dtype=np.float32)
        self.y_sample = np.array([], dtype=np.float32)
        self.kernel = kernel.with_bounds([p.range for p in params])
        self.params = params
        self.optimize_kernel = optimize_kernel
        self.gp_noise = gp_noise
        self.acquisition_function = acquisition_function

    def next(self) -> Dict[str, Union[int, float]]:
        bounds = [b.range for b in self.params]

        if len(self.X_sample) == 0:
            x_next = default_from_bounds(bounds)
        else:
            gp = self.create_gp()

            x_next = propose_location(self.acquisition_function, gp, self.y_sample.max(), bounds)

        typed_vals = [int(x) if p.range.type == "int" else x
                      for x, p in zip(x_next, self.params)]

        names = [p.name for p in self.params]
        params_dict = dict(zip(names, typed_vals))

        return params_dict

    def add_sample(self, x_next: np.ndarray, y_next: float) -> None:
        assert type(y_next) == float, f"f(x) must return a float, got type {type(y_next)}, value: {y_next}"

        if len(self.X_sample) == 0 and len(self.y_sample) == 0:
            self.X_sample = np.array([x_next])
            self.y_sample = np.array([y_next])
        else:
            self.X_sample = np.vstack((self.X_sample, x_next))
            self.y_sample = np.hstack((self.y_sample, y_next))

    def create_gp(self) -> GaussianProcess:
        gp = GaussianProcess(kernel=self.kernel, noise=self.gp_noise)
        gp.fit(self.X_sample, self.y_sample)

        if self.optimize_kernel:
            gp = gp.optimize_kernel()

        return gp

    def result(self) -> OptimizationResult:
        X_sample = self.X_sample
        y_sample = self.y_sample
        max_y_ind = self.y_sample.argmax()

        return OptimizationResult(
            X_sample,
            y_sample,
            best_x=X_sample[max_y_ind],
            best_y=y_sample[max_y_ind],
            params=self.params,
            kernel=self.kernel.copy(),
            n_iter=-1, # TODO: figure out a good way to pass this in
            opt_fun=None
        )

    def run(self, experiment, n_iter) -> None:
        done = 0
        while done < n_iter:
            params_dict = self.next()

            done += 1
            job = experiment.runner.start(params_dict)

            # TODO: !!! :D
            # loop.add_sample(np.array(list(x_next.values())), y_next)

            while not job.is_finished():
                psutil.wait_procs(psutil.Process().children(), timeout=0.01)
                time.sleep(1)


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


def dict_values(d: Dict) -> np.ndarray:
    return np.array(list(d.values()))


def bo_maximize_loop(
    f: Callable[[np.array], float],
    bounds: List[Hyperparameter],
    kernel: Kernel = SquaredExp(),
    acquisition_function=expected_improvement,
    gp_noise: float = 0,
    n_iter: int = 8,
    optimize_kernel=True,
    use_tqdm=True,
    callback: Callable = None
) -> OptimizationResult:
    loop = OptimizationLoop(bounds, kernel, optimize_kernel,
                            gp_noise, acquisition_function)

    if use_tqdm:
        from tqdm import tqdm
        iter = tqdm(range(n_iter))
    else:
        iter = range(n_iter)

    for i in iter:
        x_next = loop.next()
        y_next = f(x_next)

        x_next_values = dict_values(x_next)

        loop.add_sample(x_next_values, y_next)

        if callback is not None:
            callback(i, acquisition_function, loop.create_gp(), loop.X_sample,
                     loop.y_sample, x_next_values, y_next)

    return loop.result()


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

    return bo_maximize_loop(f, bounds, kernel, acquisition_function,
                            gp_noise=gp_noise, n_iter=n_iter,
                            callback=plot_iteration, optimize_kernel=optimize_kernel)


def plot_2d_optim_result(result: OptimizationResult, resolution: float = 30):
    # assert len(result.params) == 2

    b1 = result.params[0].range
    b2 = result.params[1].range

    x1 = np.linspace(b1.low, b1.high, resolution)
    x2 = np.linspace(b2.low, b2.high, resolution)

    assert len(x1) < 80, f"too large x1, len = {len(x1)}"
    assert len(x2) < 80, f"too large x1, len = {len(x2)}"

    gx, gy = np.meshgrid(x1, x2)

    X_2d = np.c_[gx.ravel(), gy.ravel()]

    bounds = [p.range for p in result.params]
    # TODO: optimize kernel

    X_sample = result.X_sample[:, :2]

    mu, _ = GaussianProcess(kernel=result.kernel.with_bounds(bounds)) \
        .fit(X_sample, result.y_sample).posterior(X_2d).mu_std()

    mu_mat = mu.reshape(gx.shape[0], gx.shape[1])
    extent = [b1.low, b1.high, b2.high, b2.low]

    assert result.best_x is not None

    plt.title(f"GP posterior {result.best_y:.3f}, {result.kernel}")
    plt.imshow(mu_mat, extent=extent, aspect="auto")
    plt.scatter(X_sample[:, 0], X_sample[:, 1], c="k")
    plt.scatter([result.best_x[0]], [result.best_x[1]], c="r")

    print(X_sample[0,0], X_sample[0, 1], result.y_sample[0])

    return mu_mat, extent, x1, x2
