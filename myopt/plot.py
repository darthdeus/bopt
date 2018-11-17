import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.random import multivariate_normal


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


def plot_gp(mu, cov, X, X_train=None, y_train=None, kernel=None, num_samples=3, figsize=(7, 3), figure=True):
    std = 2 * np.sqrt(np.diag(cov))  # 1.96?

    if figure:
        plt.figure(figsize=figsize)

    if kernel is not None:
        plt.title(kernel)

    plt.fill_between(X, mu + std, mu - std, alpha=0.1)
    plt.plot(X, mu, label="Mean")

    samples = multivariate_normal(mu, cov, size=num_samples)
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.7, ls="--", label=f"Sample {i+1}", color="black")

    if X_train is not None:
        plt.plot(X_train, y_train, "rx", lw=2)

    plt.legend()


def plot_approximation(ax, ei_y, X, y, gp, X_sample, y_sample, X_next=None, show_legend=False):
    mu, std = gp.posterior(X).mu_std()

    ax.fill_between(X.ravel(),
                    mu.ravel() + 1.96 * std,
                    mu.ravel() - 1.96 * std,
                    alpha=0.1)
    l1 = ax.plot(X, y, 'g--', lw=1, label='Objective')
    l2 = ax.plot(X, mu, 'b-', lw=1, label='GP mean')
    l3 = ax.plot(X_sample, y_sample, 'kx', mew=3, label='Samples')
    if X_next:
        ax.axvline(x=X_next, ls='--', c='k', lw=1)

    ax2 = ax.twinx()

    l4 = ax2.plot(X, ei_y, 'r-', lw=1, label='Acquisition fn')
    ax2.axvline(x=X_next, ls='--', c='k', lw=1)

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
