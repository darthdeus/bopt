import yaml
import os
import re
import psutil
import time
import pathlib
import datetime
import logging
import traceback

import numpy as np

from typing import List, Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bopt.basic_types import Hyperparameter, JobStatus
from bopt.job_params import JobParams
from bopt.model_config import ModelConfig
from bopt.models.model import Model
from bopt.sample import Sample, SampleCollection
from bopt.models.parameters import ModelParameters
from bopt.models.random_search import RandomSearch
from bopt.runner.abstract import Job, Runner
from bopt.runner.runner_loader import RunnerLoader

from bopt.acquisition_functions.acquisition_functions import AcquisitionFunction

# TODO: set this at a proper global place
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("GP").setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger("matplotlib").setLevel(logging.INFO)


black_cmap = LinearSegmentedColormap.from_list("black", ["black", "black"])


class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True


class Experiment:
    kernel_names = ["rbf", "Mat32", "Mat52"]
    acquisition_fn_names = ["ei", "pi"]

    hyperparameters: List[Hyperparameter]
    runner: Runner
    samples: List[Sample]
    result_regex: str

    def __init__(self, hyperparameters: List[Hyperparameter], runner: Runner, result_regex: str):
        self.hyperparameters = hyperparameters
        self.runner = runner
        self.samples = []
        self.result_regex = result_regex

    def to_dict(self) -> dict:
        return {
            "hyperparameters": {h.name: h.to_dict() for h in self.hyperparameters},
            "samples": [s.to_dict() for s in self.samples],
            "runner": self.runner.to_dict(),
            "result_regex": self.result_regex
        }

    @staticmethod
    def from_dict(data: dict) -> "Experiment":
        hyperparameters = \
            [Hyperparameter.from_dict(key, data["hyperparameters"][key])
            for key in data["hyperparameters"].keys()]

        if data["samples"] and len(data["samples"]) > 0:
            samples = [Sample.from_dict(s, hyperparameters)
                    for s in data["samples"]]
        else:
            samples = []

        runner = RunnerLoader.from_dict(data["runner"])

        experiment = Experiment(hyperparameters, runner, data["result_regex"])
        experiment.samples = samples

        return experiment

    def collect_results(self) -> None:
        for sample in self.samples:
            if sample.result is None and sample.job and sample.job.is_finished():
                # Sine we're using `handle_cd` we always assume the working directory
                # is where meta.yml is.
                fname = os.path.join("output", f"job.o{sample.job.job_id}")

                if os.path.exists(fname):
                    sample.job.finished_at = datetime.datetime.now()

                    with open(fname, "r") as f:
                        contents = f.read().rstrip("\n")

                        for line in contents.split("\n"):
                            matches = re.match(self.result_regex, line)
                            if matches:
                                sample.result = float(matches.groups()[0])

                        logging.error("Job {} seems to have failed, it finished running and its result cannot be parsed.".format(sample.job.job_id))
                else:
                    logging.error("Output file not found for job {} even though it finished. It will be considered as a failed job.".format(sample.job.job_id))

    def get_xy(self, meta_dir: str):
        samples = self.ok_samples()

        sample_col = SampleCollection(samples, meta_dir)
        X_sample, Y_sample = sample_col.to_xy()

        return X_sample, Y_sample

    def suggest(self, model_config: ModelConfig, meta_dir: str) -> Tuple[JobParams, Model]:
        # TODO: overit, ze by to fungovalo i na ok+running a mean_pred
        if len(self.ok_samples()) == 0:
            logging.info("No existing samples found, overloading suggest with RandomSearch.")
            model = RandomSearch()

            job_params, fitted_model = \
                    model.predict_next(self.hyperparameters)
        else:
            from bopt.models.gpy_model import GPyModel

            X_sample, Y_sample = self.get_xy(meta_dir)

            job_params, fitted_model = \
                    GPyModel.predict_next(model_config, self.hyperparameters, X_sample, Y_sample)

        return job_params, fitted_model

    def run_next(self, model_config: ModelConfig, meta_dir: str) -> Tuple[Job, Model, np.ndarray]:
        job_params, fitted_model = self.suggest(model_config, meta_dir)

        job, next_sample = self.manual_run(model_config, meta_dir, job_params,
                fitted_model.to_model_params())

        return job, fitted_model, next_sample.to_x()

    def manual_run(self, model_config: ModelConfig, meta_dir: str, job_params: JobParams,
            model_params: ModelParameters) -> Tuple[Job, Sample]:
        assert isinstance(job_params, JobParams)

        output_dir_path = pathlib.Path(meta_dir) / "output"
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logging.info("Output set to {}, absolute path: {}".format(output_dir_path, output_dir_path.absolute()))

        job_params.validate()

        output_dir = str(output_dir_path)

        job = self.runner.start(output_dir, job_params)

        X_sample, Y_sample = self.get_xy(meta_dir)

        if len(X_sample) > 0:
            from bopt.models.gpy_model import GPyModel

            if model_params.can_predict_mean():
                # Use the fitted model to predict mu/sigma.
                gpy_model = GPyModel.from_model_params(model_params, X_sample, Y_sample)
                model = gpy_model.model

            else:
                model = GPyModel.gpy_regression(model_config, X_sample, Y_sample)

            X_next = np.array([job_params.x])

            mu, var = model.predict(X_next)
            sigma = np.sqrt(var)

            assert mu.size == 1
            assert sigma.size == 1
        else:
            mu = 0.0
            sigma = 1.0 # TODO: lol :)

        next_sample = Sample(job, model_params, float(mu), float(sigma))
        self.samples.append(next_sample)

        self.serialize(meta_dir)
        logging.debug("Serialization done")

        return job, next_sample

    def run_single(self, model_config: ModelConfig, meta_dir: str) -> Job:
        job, fitted_model, x_next = self.run_next(model_config, meta_dir)

        # TODO: nechci radsi JobParams?
        logging.debug("Starting to plot")

        try:
            self.plot_current(fitted_model, meta_dir, x_next)
        except ValueError as e:
            traceback_str = "".join(traceback.format_tb(e.__traceback__))
            logging.error("Plotting failed, error:\n{}\n\n".format(e, traceback_str))

        logging.debug("Plotting done")

        return job

    def run_loop(self, model_config: ModelConfig, meta_dir: str,
            n_iter=10, n_parallel=1) -> None:

        n_started = 0

        while n_started < n_iter:
            if self.num_running() < n_parallel:
                self.collect_results()

                job = self.run_single(model_config, meta_dir)

                n_started += 1

                self.serialize(meta_dir)

                logging.info("Started a new job {} with config {}".format(job.job_id, model_config))

            psutil.wait_procs(psutil.Process().children(), timeout=0.01)
            time.sleep(1.0)


        # for i in range(n_iter):
        #     job = self.run_single(model_config, meta_dir)
        #     logging.info("Started a new job {} with config {}".format(job.job_id, model_config))
        #
        #     start_time = time.time()
        #
        #     # psutil.wait_procs(psutil.Process().children(), timeout=0.01)
        #     while not job.is_finished():
        #         psutil.wait_procs(psutil.Process().children(), timeout=0.01)
        #         time.sleep(0.2)
        #
        #     end_time = time.time()
        #
        #     logging.info("Job {} finished after {}".format(job.job_id, end_time - start_time))
        #
        #     self.collect_results()
        #     self.serialize(meta_dir)

    def serialize(self, meta_dir: str) -> None:
        dump = yaml.dump(self.to_dict(), default_flow_style=False, Dumper=NoAliasDumper)

        with open(os.path.join(meta_dir, "meta.yml"), "w") as f:
            f.write(dump)

    @staticmethod
    def deserialize(meta_dir: str) -> "Experiment":
        with open(os.path.join(meta_dir, "meta.yml"), "r") as f:
            contents = f.read()
            obj = yaml.load(contents, Loader=yaml.FullLoader)

        experiment = Experiment.from_dict(obj)
        experiment.collect_results()
        experiment.serialize(meta_dir)

        return experiment

    def ok_samples(self) -> List[Sample]:
        return [s for s in self.samples if s.result]

    def num_running(self) -> int:
        return len([s for s in self.samples if s.status() == JobStatus.RUNNING])

    def plot_current(self, gpy_model: Model, meta_dir: str,
            x_next: np.ndarray, resolution: float = 30) -> None:
        lows        = [h.range.low for h in self.hyperparameters]
        highs       = [h.range.high for h in self.hyperparameters]
        plot_limits = [lows, highs]

        if isinstance(gpy_model, RandomSearch):
            return

        assert gpy_model is not None, "gp is None"

        model = gpy_model.model

        param_str = "ls: {:.3f}, variance: {:.3f}, noise: {:.3f}".format(
            float(model.kern.lengthscale),
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
                self.hyperparameters, outer_grid, fig,
                gpy_model.acquisition_fn)

        plot_dir = os.path.join(meta_dir, "plots")
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        plot_fname = "opt-plot-{}.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        plot_fig_fname = os.path.join(plot_dir, plot_fname)

        plt.savefig(plot_fig_fname)


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
    from GPy.plotting.gpy_plot.plot_util import x_frame2D

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
            gs[i] = g1.copy()
            gs[i][:] = x_slice[i] # TODO: neni tohle spatne typ?

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
