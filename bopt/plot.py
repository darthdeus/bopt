from typing import List

import io
import base64
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from numpy.random import multivariate_normal
from bopt.kernels import Kernel
import bopt.kernel_opt as kernel_opt


def base64_plot():
    image = io.BytesIO()
    plt.tight_layout()
    plt.savefig(image, format='png')
    plt.gcf().clear()
    plt.close()
    image.seek(0)
    return base64.encodebytes(image.getvalue()).decode("ascii")


def imshow(data: np.ndarray, a_values: np.ndarray, b_values: np.ndarray,
           xlabel="sigma", ylabel="lengthscale", title="Kernel marginal likelihood"):
    plt.figure(figsize=(5,5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    im = plt.imshow(data, extent=[min(b_values), max(b_values), max(a_values), min(a_values)], aspect="auto")
    plt.colorbar(im)
    plt.title(title)


def plots(*plots, n_row=3, figsize=(15, 4)):
    num_rows = len(plots) // n_row + 1

    plt.figure(figsize=figsize)

    for i, plot in enumerate(plots):
        plt.subplot(num_rows, n_row, i + 1)
        plt.imshow(plot)

    plt.show()


def plot_gp_prior(mu, cov, X, kernel=None, num_samples=3):
    std = 2 * np.sqrt(np.diag(cov))  # 1.96?

    if kernel is not None:
        plt.title(kernel)

    plt.fill_between(X, mu + std, mu - std, alpha=0.1)
    plt.plot(X, mu, label="Mean")

    samples = multivariate_normal(mu, cov, size=num_samples)
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.7, ls="--", label=f"Sample {i+1}", color="black")


def plot_gp(mu, cov, X, X_train=None, y_train=None, kernel=None, noise: int = 0,
        num_samples=3, figsize=(7, 3), figure=True, nll=None):
    assert X.ndim == 1

    std = 2 * np.sqrt(np.diag(cov))

    if figure:
        plt.figure(figsize=figsize)

    if kernel is not None:
        # TODO: fuj duplicity
        if nll is not None:
            plt.title("Noise: {:.3f}, {}, nll={}".format(noise, kernel, nll))
        else:
            plt.title("Noise: {:.3f}, {}".format(noise, kernel))

    plt.fill_between(X, mu + std, mu - std, alpha=0.1)
    plt.plot(X, mu, label="Mean")

    samples = multivariate_normal(mu, cov, size=num_samples)
    # print(mu.mean(), cov.mean(), noise)

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.7, ls="--",
                label=f"Sample {i+1}", color="black")

    if X_train is not None:
        assert X_train.ndim == 1
        assert y_train.ndim == 1
        plt.plot(X_train, y_train, "rx", lw=2)

    plt.legend()

# def plot_gp(mu, cov, X, X_train=None, y_train=None, kernel=None, num_samples=3, figsize=(7, 3), figure=True):
#     std = 2 * np.sqrt(np.diag(cov))  # 1.96?
#
#     if figure:
#         plt.figure(figsize=figsize)
#
#     if kernel is not None:
#         plt.title(kernel)
#
#     X = X
#     X_train = X_train[:4]
#     y_train = y_train[:4]
#     mu = mu
#
#     # plt.fill_between(X, mu + std, mu - std, alpha=0.1)
#     plt.plot(X, mu, label="Mean")
#
#     # samples = multivariate_normal(mu, cov, size=num_samples)
#     # for i, sample in enumerate(samples):
#     #     plt.plot(X, sample, lw=0.7, ls="--", label=f"Sample {i+1}", color="black")
#
#     if X_train is not None:
#         plt.plot(X_train, y_train, "rx", lw=2)
#
#     plt.legend()
#


def plot_approximation(ax, ei_y, X, y, gp, X_sample, y_sample, multiple_x_next: List[np.ndarray], show_legend=False):
    mu, std = gp.posterior(X).mu_std()

    ax.fill_between(X.ravel(),
                    mu.ravel() + 1.96 * std,
                    mu.ravel() - 1.96 * std,
                    alpha=0.1)
    l1 = ax.plot(X, y, 'g--', lw=1, label='Objective')
    l2 = ax.plot(X, mu, 'b-', lw=1, label='GP mean')
    l3 = ax.plot(X_sample, y_sample, 'kx', mew=3, label='Samples')

    for x_next in multiple_x_next:
        ax.axvline(x=x_next, ls='--', c='k', lw=1)

    ax2 = ax.twinx()

    l4 = ax2.plot(X, ei_y, 'r-', lw=1, label='Acquisition fn')

    for x_next in multiple_x_next:
        ax2.axvline(x=x_next, ls='--', c='k', lw=1)

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]

    if show_legend:
        plt.legend(lns, labs)


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


def plot_gp_2D(gx, gy, mu, X_train, y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c=y_train, cmap=cm.coolwarm)
    ax.set_title(title)


def plot_kernel_loss(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray,
                     noise_level: float = 0.1, xmax: int = 5, sigma: float = 1) -> None:
    X = np.linspace(0.00001, xmax, num=50)

    def likelihood(l):
        return kernel_opt.kernel_log_likelihood(
                 kernel.set_params(np.array([l, sigma])),
                 X_train, y_train, noise_level)

    data = np.vectorize(likelihood)(X)

    plt.plot(X, data)
    plt.title(f"Kernel marginal likelihood, $\\sigma = {sigma}$")


def plot_kernel_loss_2d(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray,
                        noise_level: float = 0.1) -> None:
    num_points = 10
    data = np.zeros((num_points, num_points))

    amin = 0.3
    amax = 5

    bmin = 1
    bmax = 10

    a_values = np.linspace(amin, amax, num=num_points)
    b_values = np.linspace(bmin, bmax, num=num_points)

    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            theta = np.array([a, b])
            data[i, j] = kernel_opt.kernel_log_likelihood(kernel.set_params(theta), X_train, y_train, noise_level)

    imshow(data, a_values, b_values)

