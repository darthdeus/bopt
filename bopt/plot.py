from typing import List

import datetime
import os
import io
import base64

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from bopt.experiment import Experiment
from bopt.acquisition_functions.acquisition_functions import AcquisitionFunction
from bopt.models.random_search import RandomSearch
from bopt.models.gpy_model import GPyModel
from bopt.models.model import Model

black_cmap = LinearSegmentedColormap.from_list("black", ["black", "black"])


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


def plot_current(experiment: Experiment, gpy_model: Model,
        x_next: np.ndarray, resolution: float = 30) -> None:
    lows        = [h.range.low for h in experiment.hyperparameters]
    highs       = [h.range.high for h in experiment.hyperparameters]
    plot_limits = [lows, highs]

    if isinstance(gpy_model, RandomSearch):
        return

    assert isinstance(gpy_model, GPyModel)

    assert gpy_model is not None, "gp is None"

    model = gpy_model.model

    param_str = "ls: {}, variance: {:.3f}, noise: {:.3f}".format(
        model.kern.lengthscale,
        float(model.kern.variance),
        float(model.Gaussian_noise.variance)
    )

    vmin = model.Y.min()
    vmax = model.Y.max()

    # TODO: funkcni stary plotovani
    # ax = plt.subplot(4, 1, 4)
    # plot_limits = [[gx.min(), gy.min()], [gx.max(), gy.max()]]
    # model.plot_mean(ax=ax, vmin=vmin, vmax=vmax, plot_limits=plot_limits, cmap="jet", label="Mean")
    #
    # black_cmap = LinearSegmentedColormap.from_list("black", ["black", "black"])
    # model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10, s=60)

    # TODO: new point
    # TODO: plot current max

    fig = plt.figure(figsize=(15, 15))
    outer_grid = gridspec.GridSpec(1, 1)

    max_idx = model.Y.reshape(-1).argmax()
    x_max = model.X[max_idx]

    # with plt.xkcd():
    plot_objective(model, x_max, x_next, plot_limits, vmin, vmax,
            experiment.hyperparameters, outer_grid, fig,
            gpy_model.acquisition_fn)

    plot_dir = "plots"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    plot_fname = "opt-plot-{}.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    plot_fig_fname = os.path.join(plot_dir, plot_fname)

    plt.savefig(plot_fig_fname)
    plt.close()


def plot_objective(model, x_slice, x_next, plot_limits, vmin, vmax,
        hyperparameters, outer_grid, fig,
        acq, zscale='linear', dimensions=None) -> None:
    """
    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.
    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.
    """

    # Number of hyperparameters
    n_dims = model.X.shape[1]

    inner_grid = gridspec.GridSpecFromSubplotSpec(n_dims, n_dims, subplot_spec=outer_grid[0, 0],
            hspace=0.2, wspace=0.2)
    # second_grid = gridspec.GridSpecFromSubplotSpec(n_dims, n_dims, subplot_spec=outer_grid[1, 0])

    # fig, ax = plt.subplots(n_dims, n_dims, figsize=(size * n_dims, size * n_dims))
    plt.suptitle(str(list(map(lambda x: str(round(x, 3)),
        model.param_array.tolist()))) + " " + str(x_slice.tolist()))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(n_dims):
        for j in range(n_dims):
            ax = plt.Subplot(fig, inner_grid[i * n_dims + j])
            # ax2 = plt.Subplot(fig, second_grid[i * n_dims + j])

            if i == j:
                fixed_inputs = []
                for f in list(set(range(n_dims)) - set([i])):
                    fixed_inputs.append((f, x_slice[f].item()))

                model.plot_mean(ax=ax, fixed_inputs=fixed_inputs,
                        plot_limits=[p[i] for p in plot_limits])
                ax.set_xlabel(hyperparameters[i].name)

                model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10,
                        s=60, visible_dims=[i])


            #     xi, yi = partial_dependence(space, result.models[-1], i,
            #                                 j=None,
            #                                 sample_points=rvs_transformed,
            #                                 n_points=n_points)
            #
                ax.axvline(x_next[i], linestyle="--", color="r", lw=1)

            # lower triangle
            else:
                fixed_inputs = []
                for f in list(set(range(n_dims)) - set([i, j])):
                    fixed_inputs.append((f, x_slice[f]))

                lims = [np.array(p)[[i, j]] for p in plot_limits]

                if i > j:
                    # TODO: plot vs plot_mean
                    # TODO: neplotovat data 2x
                    model.plot(ax=ax, fixed_inputs=fixed_inputs, cmap="jet",
                            vmin=vmin, vmax=vmax, plot_limits=lims, legend=False)

                    ax.set_xlabel(hyperparameters[i].name)
                    ax.set_ylabel(hyperparameters[j].name)

                    model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10,
                            s=60, visible_dims=[i, j])

                    ax.axvline(x_next[i], linestyle="--", color="r", lw=1)
                    ax.axhline(x_next[j], linestyle="--", color="r", lw=1)

                elif i < j:
                    acq_ax = ax

                    acq_for_dims(model, acq, x_next, hyperparameters, acq_ax, [i, j], lims)
                    model.plot_data(ax=acq_ax, alpha=1, cmap=black_cmap,
                            zorder=10, s=60, visible_dims=[i, j])

                    acq_ax.axvline(x_next[i], linestyle="--", color="r", lw=1)
                    acq_ax.axhline(x_next[j], linestyle="--", color="r", lw=1)

                    # xi, yi, zi = partial_dependence(space, result.models[-1],
                    #                                 i, j,
                    #                                 rvs_transformed, n_points)
                    # ax[i, j].contourf(xi, yi, zi, levels,
                    #                   locator=locator, cmap='viridis_r')
                    # ax[i, j].scatter(samples[:, j], samples[:, i],
                    #                  c='k', s=10, lw=0.)
                    # ax[i, j].scatter(result.x[j], result.x[i],
                    #                  c=['r'], s=20, lw=0.)

            fig.add_subplot(ax)
            # fig.add_subplot(ax2)

    # return _format_scatter_plot_axes(ax, space, ylabel="Partial dependence",
    #                                  dim_labels=dimensions)



def acq_for_dims(model, acq: AcquisitionFunction, x_slice, hyperparameters, ax,
        dims, plot_limits) -> None:
    # x_frame2D only checks shape[1] and not the data when `plot_limits` is provided
    frame_shaped_array = np.zeros((1,2))

    resolution = 50

    d1 = np.linspace(plot_limits[0][0], plot_limits[1][0], num=resolution)
    d2 = np.linspace(plot_limits[0][1], plot_limits[1][1], num=resolution)

    g1, g2 = np.meshgrid(d1, d2)

    gs = [0] * len(x_slice)
    gs[dims[0]] = g1
    gs[dims[1]] = g2

    for i in range(len(x_slice)):
        if i not in dims:
            gs[i] = np.full(g1.shape, x_slice[i])

    grid = np.stack(gs, axis=-1)

    mu, var = model.predict(grid.reshape(resolution * resolution, -1))
    sigma = np.sqrt(var)

    ei = acq.raw_call(mu, sigma, model.Y.max())

    ei_mat = ei.reshape(resolution, resolution)
    # TODO: fuj, contour not increasing?
    ei_mat += ei_mat.mean()

    ax.set_xlim(left=plot_limits[0][0], right=plot_limits[1][0])
    ax.set_ylim(bottom=plot_limits[0][1], top=plot_limits[1][1])
    ax.contour(g1, g2, ei_mat, cmap="jet")# , vmin=0, vmax=500)
    # ax.colorbar()

    # fixed_inputs = [[0, X_sample[-1][0]]]
    # model.plot(ax=ax, fixed_inputs=fixed_inputs, plot_limits=[p[0] for p in plot_limits], legend=False)
    # ax.set_xlabel(self.hyperparameters[0].name)
    #
    # model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10, s=60, visible_dims=[0])
