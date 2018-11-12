import functools
from collections import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .gaussian_process import GaussianProcess
from .plot import plot_approximation, plot_convergence
from .acquisition_functions import expected_improvement
from .kernels import SquaredExp


def bo_minimize(f: Callable, noise: float, bounds: np.ndarray,
                X_init: np.ndarray, y_init: np.ndarray,
                X_true: np.ndarray = None, y_true: np.ndarray= None,
                kernel=SquaredExp(),
                acquisition_function=expected_improvement,
                n_iter: int=8, plot=True, plot_every: int=1, optimize_kernel=False):

    num_plots = (n_iter // plot_every)

    if plot:
        plt.figure(figsize=(14, num_plots * 2))
        plt.subplots_adjust(hspace=0.4)

    X_sample = X_init
    y_sample = y_init

    for i in range(n_iter):
        gp = GaussianProcess(kernel=kernel).fit(X_sample, y_sample)
        if optimize_kernel:
            gp = gp.optimize_kernel()

        bound_acquisition_function = functools.partial(acquisition_function, gp=gp, X_sample=X_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(bound_acquisition_function, X_sample, bounds)

        # Obtain next noisy sample from the objective function
        y_next = f(X_next, noise)

        ei_y = bound_acquisition_function(X_true)

        if plot:
            if i % plot_every == 0:
                # Plot samples, surrogate function, noise-free objective and next sampling location
                ax1 = plt.subplot(num_plots // 2 + 1, 2, i // plot_every + 1)

                plot_approximation(ax1, ei_y, gp.kernel, X_true, y_true, X_sample, y_sample, X_next, show_legend=i == 0)

                plt.title(f'Iteration {i+1}, {gp.kernel}')

        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))

    if plot:
        plot_convergence(X_sample, y_sample)


def propose_location(acquisition, X_sample, bounds, n_restarts=25,):
    """
    Proposes the next sampling point by optimizing the acquisition function.

    Args:

    acquisition: Acquisition function.
    X_sample: Sample locations (n x d).
    y_sample: Sample values (n x 1).

    Returns: Location of the acquisition function maximum.
    """
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim))

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)