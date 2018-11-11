import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal


def plots(*plots, n_row=3, figsize=(15, 4)):
    num_rows = len(plots) // n_row + 1

    plt.figure(figsize=figsize)

    for i, plot in enumerate(plots):
        plt.subplot(num_rows, n_row, i + 1)
        plt.imshow(plot)

    plt.show()


def plot_gp_prior(mu, cov, X, num_samples=3):
    std = 2 * np.sqrt(np.diag(cov))  # 1.96?
    plt.fill_between(X, mu + std, mu - std, alpha=0.1)
    plt.plot(X, mu, label="Mean")

    samples = multivariate_normal(mu, cov, size=num_samples)
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.7, ls="--", label=f"Sample {i+1}", color="black")


def plot_gp(mu, cov, X, X_train=None, y_train=None, num_samples=3, figsize=(7, 4), figure=True):
    std = 2 * np.sqrt(np.diag(cov))  # 1.96?

    if figure:
        plt.figure(figsize=figsize)

    plt.fill_between(X, mu + std, mu - std, alpha=0.1)
    plt.plot(X, mu, label="Mean")

    samples = multivariate_normal(mu, cov, size=num_samples)
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.7, ls="--", label=f"Sample {i+1}", color="black")

    if X_train is not None:
        plt.plot(X_train, y_train, "rx", lw=2)

    plt.legend()