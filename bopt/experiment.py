import yaml
import os
import psutil
import time
import pathlib
import numpy as np

from typing import List, Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import GPy

from bopt.models.model import Model
from bopt.sample import Sample, SampleCollection
from bopt.models.model_loader import ModelLoader
from bopt.models.parameters import ModelParameters
from bopt.models.random_search import RandomSearch
from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job, Runner
from bopt.runner.runner_loader import RunnerLoader

from bopt.optimization_result import OptimizationResult
from bopt.acquisition_functions.acquisition_functions import expected_improvement_f


black_cmap = LinearSegmentedColormap.from_list("black", ["black", "black"])


class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True


# TODO: fix numpy being stored everywhere when serializing! possibly in hyp.sample? :(
class Experiment:
    hyperparameters: List[Hyperparameter]
    runner: Runner
    samples: List[Sample]
    last_model: Optional[ModelParameters]

    def __init__(self, hyperparameters: List[Hyperparameter], runner: Runner):
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.samples = []
        self.last_model = None

    def to_dict(self) -> dict:
        return {
            "hyperparameters": {h.name: h.to_dict() for h in self.hyperparameters},
            "samples": [s.to_dict() for s in self.samples],
            "runner": self.runner.to_dict(),
            # TODO: last_model
            # "last_model": self.last_model.to_dict() if self.last_model is not None else None
        }

    @staticmethod
    def from_dict(data: dict) -> "Experiment":
        hyperparameters = \
            [Hyperparameter.from_dict(key, data["hyperparameters"][key])
            for key in data["hyperparameters"].keys()]

        samples = [Sample.from_dict(s) for s in data["samples"]]
        runner = RunnerLoader.from_dict(data["runner"])
        # last_model = ModelLoader.from_dict(data["last_model"])

        experiment = Experiment(hyperparameters, runner)
        experiment.samples = samples
        # experiment.last_model = last_model

        return experiment

    def run_next(self, model: Model, meta_dir: str, output_dir: str) -> Tuple[Job, Model, np.ndarray]:
        if len(self.samples) == 0:
            model = RandomSearch()

        sample_collection = SampleCollection(self.ok_samples(), meta_dir)

        # TODO: pridat normalizaci
        next_params, fitted_model = \
                model.predict_next(self.hyperparameters, sample_collection)

        job = self.runner.start(output_dir, next_params)

        next_sample = Sample(job, fitted_model.to_model_params())

        self.samples.append(next_sample)

        self.last_model = next_sample.model

        return job, fitted_model, next_sample.to_x()

    def run_loop(self, model: Model, meta_dir: str, n_iter=20) -> None:
        output_dir = pathlib.Path(meta_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_iter):
            job, fitted_model, x_next = self.run_next(model, meta_dir, str(output_dir))

            ##################################
            # TODO: plot with GPy!!!
            ##################################
            self.plot_current(fitted_model, meta_dir, x_next)

            while not job.is_finished():
                psutil.wait_procs(psutil.Process().children(), timeout=0.01)
                time.sleep(0.2)

            self.serialize(meta_dir)

            # optim_result = self.current_optim_result(meta_dir)
            #
            # # TODO: used the job model
            # # TODO: plot optim from job?
            # gp = optim_result.fit_gp()
            #
            # from bopt.bayesian_optimization import plot_2d_optim_result
            #
            # self.plot_current(meta_dir)

    def serialize(self, meta_dir: str) -> None:
        dump = yaml.dump(self.to_dict(), default_flow_style=False, Dumper=NoAliasDumper)

        with open(os.path.join(meta_dir, "meta.yml"), "w") as f:
            f.write(dump)

    @staticmethod
    def deserialize(meta_dir: str) -> "Experiment":
        with open(os.path.join(meta_dir, "meta.yml"), "r") as f:
            contents = f.read()
            obj = yaml.load(contents)

        return Experiment.from_dict(obj)

    def ok_samples(self) -> List[Sample]:
        return [s for s in self.samples if s.job.is_finished()]

    def current_optim_result(self, meta_dir: str) -> OptimizationResult:
        sample_col = SampleCollection(self.ok_samples(), meta_dir)

        X_sample, y_sample = sample_col.to_xy()

        # TODO: this should be handled better
        params = sorted(self.hyperparameters, key=lambda h: h.name)

        # TODO; normalizace?
        y_sample = (y_sample - y_sample.mean()) / y_sample.std()

        best_y = None
        best_x = None

        if len(y_sample) > 0:
            best_idx = np.argmax(y_sample)
            best_y = y_sample[best_idx]
            best_x = X_sample[best_idx]

        # TODO: fuj
        from bopt.kernels.kernels import SquaredExp

        kernel = SquaredExp()
        n_iter = len(X_sample)

        return OptimizationResult(
                X_sample,
                y_sample,
                best_x,
                best_y,
                params,
                kernel,
                n_iter,
                opt_fun=None)


    def plot_current(self, model: Model, meta_dir: str, x_next: np.ndarray, resolution: float = 30):
        # TODO: handle more than 2 dimensions properly
        # assert len(result.params) == 2

        lows        = [h.range.low for h in self.hyperparameters]
        highs       = [h.range.high for h in self.hyperparameters]
        plot_limits = [lows, highs]

        # TODO: float64
        # x1 = np.linspace(b1.low, b1.high, resolution, dtype=np.float64)
        # x2 = np.linspace(b2.low, b2.high, resolution, dtype=np.float64)
        #
        # assert len(x1) < 80, f"too large x1, len = {len(x1)}"
        # assert len(x2) < 80, f"too large x1, len = {len(x2)}"
        #
        # gx, gy = np.meshgrid(x1, x2)
        #
        # X_2d = np.c_[gx.ravel(), gy.ravel()]
        #
        # bounds = [p.range for p in self.hyperparameters]

        # TODO: nepotrebuju?
        # X_sample, y_sample = SampleCollection(self.samples, meta_dir).to_xy()
        #
        # np.save("tmp/data_X", X_sample)
        # np.save("tmp/data_Y", y_sample)

        if isinstance(model, RandomSearch):
            return

        assert model is not None, "gp is None"

        # TODO: with_bounds?
        # gp = GaussianProcess(kernel=result.kernel.with_bounds(bounds)) \
        #     .fit(X_sample, result.y_sample) \
        #     .optimize_kernel()

        # TODO: lol :)
        model = model.model

        # TODO: use paramz properly
        param_str = "ls: {:.3f}, variance: {:.3f}, noise: {:.3f}".format(
                float(model.kern.lengthscale),
                float(model.kern.variance),
                float(model.Gaussian_noise.variance)

        )
        # param_str = str(gp.kern) + " " + str(gp.noise) + f" nll={round(gp.log_prob().numpy().item(), 2)}"

        # TODO: take as input
        # vmin = y_sample.min()
        # vmax = y_sample.max()
        vmin = model.Y.min()
        vmax = model.Y.max()

        import matplotlib.pyplot as plt
        import datetime
        # assert result.best_x is not None

        # plt.title(f"LBFGS={U_LB} TF={U_TF}   noise={round(gp.noise, 2)} {result.kernel}", fontsize=20)
        # plt.pcolor(mu_mat, extent=extent, aspect="auto")


        # TODO: funkcni stary plotovani
        # plt.figure(figsize=(12, 16))
        #
        # plt.suptitle(param_str, fontsize=14)
        #
        # plt.subplot(4, 1, 1)
        # plt.title("Mean")
        # plt.pcolor(gx, gy, mu_mat, cmap="jet", vmin=vmin, vmax=vmax)
        # plt.colorbar()
        # plt.scatter(X_sample[:, 0], X_sample[:, 1], c="k")
        #
        # plt.subplot(4, 1, 2)
        # plt.title("Sigma")
        # plt.pcolor(gx, gy, std_mat, cmap="jet", vmin=vmin, vmax=vmax)
        # plt.colorbar()
        # plt.scatter(X_sample[:, 0], X_sample[:, 1], c="k")
        #
        # plt.subplot(4, 1, 3)
        # plt.title("Expected Improvement")
        # plt.pcolor(gx, gy, ei_mat, cmap="jet")# , vmin=0, vmax=500)
        # plt.colorbar()
        # plt.scatter(X_sample[:, 0], X_sample[:, 1], c="k")
        # # plt.scatter([result.best_x[0]], [result.best_x[1]], c="r")
        #
        # ax = plt.subplot(4, 1, 4)
        # plot_limits = [[gx.min(), gy.min()], [gx.max(), gy.max()]]
        # model.plot_mean(ax=ax, vmin=vmin, vmax=vmax, plot_limits=plot_limits, cmap="jet", label="Mean")
        #
        # black_cmap = LinearSegmentedColormap.from_list("black", ["black", "black"])
        # model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10, s=60)

        # TODO: new point
        # TODO: plot current max
        # plt.scatter(

        # plot_limits = [[gx.min(), gy.min()], [gx.max(), gy.max()]]

        fig = plt.figure(figsize=(15, 15))
        outer_grid = gridspec.GridSpec(1, 1)

        max_idx = model.Y.reshape(-1).argmax()
        x_max = model.X[max_idx]


        # with plt.xkcd():
            # plot_objective(model, X_sample[-1], plot_limits, vmin, vmax, self.hyperparameters, outer_grid, fig)
        plot_objective(model, x_max, x_next, plot_limits, vmin, vmax, self.hyperparameters, outer_grid, fig)

            # ax = plt.subplot(outer_grid[1, 0])

            # ei_for_dims(model, X_sample, y_sample, self.hyperparameters, ax, [0, 1])


        plt.savefig("tmp/opt-plot-{}.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))


def plot_objective(model, x_slice, x_next, plot_limits, vmin, vmax, hyperparameters, outer_grid, fig,
        levels=10, n_points=40, n_samples=250, size=4, zscale='linear', dimensions=None):
    """Pairwise partial dependence plot of the objective function.
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.
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
    plt.suptitle(str(list(map(lambda x: str(round(x, 3)), model.param_array.tolist()))) + " " + str(x_slice.tolist()))

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

                model.plot(ax=ax, fixed_inputs=fixed_inputs,
                        plot_limits=[p[i] for p in plot_limits], legend=False)
                ax.set_xlabel(hyperparameters[i].name)

                model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10, s=60, visible_dims=[i])

            #
            #     xi, yi = partial_dependence(space, result.models[-1], i,
            #                                 j=None,
            #                                 sample_points=rvs_transformed,
            #                                 n_points=n_points)
            #
            #     ax[i, i].plot(xi, yi)
                ax.axvline(x_next[i], linestyle="--", color="r", lw=1)

            # lower triangle
            else:
                fixed_inputs = []
                for f in list(set(range(n_dims)) - set([i, j])):
                    fixed_inputs.append((f, x_slice[f]))

                lims = [np.array(p)[[i, j]] for p in plot_limits]

                if i > j:
                    # TODO: plot vs plot_mean
                    model.plot(ax=ax, fixed_inputs=fixed_inputs, cmap="jet", label="Mean",
                            vmin=vmin, vmax=vmax, plot_limits=lims, legend=False)
                    ax.set_xlabel(hyperparameters[i].name)
                    ax.set_ylabel(hyperparameters[j].name)

                    # TODO: vratit
                    model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10, s=60, visible_dims=[i, j])
                    ax.axvline(x_next[i], linestyle="--", color="r", lw=1)
                    ax.axhline(x_next[j], linestyle="--", color="r", lw=1)

                elif i < j:
                    ei_ax = ax

                    ei_for_dims(model, x_next, hyperparameters, ei_ax, [i, j], lims)
                    model.plot_data(ax=ei_ax, alpha=1, cmap=black_cmap, zorder=10, s=60, visible_dims=[i, j])
                    ei_ax.axvline(x_next[i], linestyle="--", color="r", lw=1)
                    ei_ax.axhline(x_next[j], linestyle="--", color="r", lw=1)

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

def ei_for_dims(model, x_slice, hyperparameters, ax, dims, plot_limits):
    # x_frame2D only checks shape[1] and not the data when `plot_limits` is provided
    from GPy.plotting.gpy_plot.plot_util import x_frame2D

    frame_shaped_array = np.zeros((1,2))

    # Xnew, xx, yy, xmin, xmax = x_frame2D(frame_shaped_array, plot_limits=plot_limits)
    #
    # X_2d_pred = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]
    resolution = 50

    d1 = np.linspace(plot_limits[0][0], plot_limits[1][0], num=resolution)
    d2 = np.linspace(plot_limits[0][1], plot_limits[1][1], num=resolution)

    g1, g2 = np.meshgrid(d1, d2)

    gs = [0] * len(x_slice)
    gs[dims[0]] = g1
    gs[dims[1]] = g2

    for i in range(len(x_slice)):
        if i not in dims:
            gs[i] = g1.copy()
            gs[i][:] = x_slice[i]

    grid = np.stack(gs, axis=-1)

    # np.stack([g1, g2, g3], axis=-1)

    # grid = np.zeros([resolution, resolution, len(x_slice)])
    #
    # for i in range(resolution):
    #     for j in range(resolution):
    #         grid[i, j, dims[0]] = x_slice[dims[0]]
    #         grid[i, j, dims[1]] = x_slice[dims[1]]



    # TODO: stare plotovani, nefunguje na vic nez 2d
    mu, var = model.predict(grid.reshape(resolution * resolution, -1))

    # TODO: std or var?
    std = np.sqrt(var)

    ei = expected_improvement_f(mu, std, model.Y.max())

    # b1 = hyperparameters[0].range
    # b2 = hyperparameters[1].range

    # TODO: fuj
    # mu_mat  = mu.reshape(xx.shape[0], xx.shape[1])
    # std_mat = std.reshape(xx.shape[0], xx.shape[1])

    ei_mat = ei.reshape(resolution, resolution)
    # extent  = [b1.low, b1.high, b2.high, b2.low]

    ei_mat += ei_mat.mean()

    ax.set_xlim(left=plot_limits[0][0], right=plot_limits[1][0])
    ax.set_ylim(bottom=plot_limits[0][1], top=plot_limits[1][1])
    # ax.pcolor(xx, yy, ei_mat, cmap="jet")# , vmin=0, vmax=500)
    ax.contour(g1, g2, ei_mat, cmap="jet")# , vmin=0, vmax=500)
    # ax.colorbar()

    # fixed_inputs = [[0, X_sample[-1][0]]]
    # model.plot(ax=ax, fixed_inputs=fixed_inputs, plot_limits=[p[0] for p in plot_limits], legend=False)
    # ax.set_xlabel(self.hyperparameters[0].name)
    #
    # model.plot_data(ax=ax, alpha=1, cmap=black_cmap, zorder=10, s=60, visible_dims=[0])
