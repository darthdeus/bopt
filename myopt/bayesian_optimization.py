import functools
from collections import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .acquisition_functions import expected_improvement
from .gaussian_process import GaussianProcess
from .kernels import SquaredExp


def bo_minimize(f: Callable, noise: float, bounds: np.ndarray,
                X_init: np.ndarray, y_init: np.ndarray,
                X_true: np.ndarray = None, y_true: np.ndarray= None,
                kernel=SquaredExp(),
                acquisition_function=expected_improvement,
                n_iter: int=7, plot=True):
    if plot:
        plt.figure(figsize=(12, n_iter * 3))
        plt.subplots_adjust(hspace=0.4)

    X_sample = X_init
    y_sample = y_init

    bound_acquisition_function = functools.partial(acquisition_function, kernel=kernel)

    for i in range(n_iter):
        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(bound_acquisition_function, X_sample, y_sample, bounds)

        # Obtain next noisy sample from the objective function
        y_next = f(X_next, noise)

        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(kernel, X_true, y_true, X_sample, y_sample, X_next, show_legend=i == 0)
        plt.title(f'Iteration {i+1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        future_location = bound_acquisition_function(X_true, X_sample, y_sample)

        plot_acquisition(X_true, future_location, X_next, show_legend=i == 0)

        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))


def plot_approximation(kernel, X, Y, X_sample, y_sample, X_next=None, show_legend=False):
    # mu, std = gp_reg(X_sample, y_sample, X, return_std=True)
    mu, std = GaussianProcess(kernel=kernel).fit(X_sample, y_sample).posterior(X).mu_std()

    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def plot_acquisition(X, y, X_next, show_legend=False):
    plt.plot(X, y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()


def plot_convergence(X_sample, y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = y_sample[n_init:].ravel()
    r = range(1, len(x) + 1)

    x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')


def propose_location(acquisition, X_sample, y_sample, bounds, n_restarts=25,):
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
        return -acquisition(X.reshape(-1, dim), X_sample, y_sample)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)