from typing import List

import io
import base64
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm


def base64_plot():
    image = io.BytesIO()
    plt.tight_layout()
    plt.savefig(image, format='png')
    plt.gcf().clear()
    plt.close()
    image.seek(0)
    return base64.encodebytes(image.getvalue()).decode("ascii")


def plots(*plots, n_row=3, figsize=(15, 4)):
    num_rows = len(plots) // n_row + 1

    plt.figure(figsize=figsize)

    for i, plot in enumerate(plots):
        plt.subplot(num_rows, n_row, i + 1)
        plt.imshow(plot)

    plt.show()


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


