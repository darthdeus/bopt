from collections import defaultdict
import math
import sys
import jsonpickle
import numpy as np
import logging

from typing import NamedTuple, List, Tuple, Dict
from livereload import Server
from flask import Flask, render_template, request
import GPy

import bopt
from bopt.cli.util import handle_cd_revertible, acquire_lock


def create_gp_for_data(experiment, hyperparameters, X, Y):
    assert X.ndim == 2
    assert 1 <= X.shape[1] <= 2

    model = GPy.models.GPRegression(X, Y, kernel=GPy.kern.Matern52(input_dim=X.shape[1], ARD=True))

    min_bound = 1e-1
    max_bound = 1e3

    # model.Gaussian_noise.variance.unconstrain()
    # model.Gaussian_noise.variance.constrain_bounded(min_bound, max_bound)
    # model.kern.variance.unconstrain()
    # model.kern.variance.constrain_bounded(min_bound, max_bound)
    # model.kern.lengthscale.unconstrain()
    # model.kern.lengthscale.constrain_bounded(min_bound, max_bound)

    gamma_a = 1.0
    gamma_b = 0.01

    # model.Gaussian_noise.variance.unconstrain()
    # model.Gaussian_noise.variance.constrain_bounded(min_bound, max_bound)

    model.kern.lengthscale.unconstrain()

    for i, param in enumerate(hyperparameters):
        prior = bopt.GPyModel.prior_for_hyperparam(experiment.gp_config, param)
        model.kern.lengthscale[[i]].set_prior(prior)

    variance_prior = GPy.priors.Gamma(experiment.gp_config.gamma_a, experiment.gp_config.gamma_b)

    model.kern.variance.unconstrain()
    model.kern.variance.set_prior(variance_prior)

    model.optimize()

    logging.info("GP hyperparams: {}".format(model.param_array.tolist()))

    return model


class Slice1D:
    param: bopt.Hyperparameter
    x: List[float]

    x_slice_at: float

    mu: List[float]
    sigma: List[float]
    acq: List[float]

    other_samples: Dict[str, List[float]]

    model: GPy.models.GPRegression

    def __init__(self, param: bopt.Hyperparameter, x: List[float],
            x_slice_at: float, mu: List[float], sigma: List[float],
            acq: List[float], other_samples: Dict[str, List[float]],
            model: GPy.models.GPRegression) -> None:
        self.param = param
        self.x = x
        self.x_slice_at = x_slice_at
        self.mu = mu
        self.sigma = sigma
        self.acq = acq
        self.other_samples = other_samples
        self.model = model

    def sigma_low(self) -> List[float]:
        return [m - s for m, s in zip(self.mu, self.sigma)]

    def sigma_high(self) -> List[float]:
        return [m + s for m, s in zip(self.mu, self.sigma)]

    def mu_bounds(self, show_acq: int) -> Tuple[float, float]:
        other_results = self.other_samples["y"]

        if show_acq == 1:
            low = min(self.sigma_low() + self.acq + other_results)
            high = max(self.sigma_high() + self.acq + other_results)
        else:
            low = min(self.sigma_low() + other_results)
            high = max(self.sigma_high() + other_results)

        return low, high

    def x_range(self) -> Tuple[float, float]:
        low = self.param.range.low
        high = self.param.range.high

        if self.param.range.is_logscale():
            low = math.log10(low)
            high = math.log10(high)

        margin = (high - low) * 0.05

        return low - margin, high + margin


class Slice2D:
    p1: bopt.Hyperparameter
    p2: bopt.Hyperparameter

    x1: List[float]
    x2: List[float]

    x1_slice_at: float
    x2_slice_at: float

    mu: List[float]

    other_samples: Dict[str, List[float]]

    model: GPy.models.GPRegression

    def __init__(self,
            p1: bopt.Hyperparameter, p2: bopt.Hyperparameter,
            x1: List[float], x2: List[float],
            x1_slice_at: float, x2_slice_at: float,
            mu: List[float], other_samples: Dict[str, List[float]],
            model: GPy.models.GPRegression) -> None:
        self.p1 = p1
        self.p2 = p2

        self.x1 = x1
        self.x2 = x2

        self.x1_slice_at = x1_slice_at
        self.x2_slice_at = x2_slice_at

        self.mu = mu

        self.other_samples = other_samples

        self.model = model

    def x1_bounds(self) -> Tuple[float, float]:
        return min(self.x1), max(self.x1)

    def x2_bounds(self) -> Tuple[float, float]:
        return min(self.x2), max(self.x2)


def create_slice_1d(i: int, experiment: bopt.Experiment, resolution: int,
        n_dims: int, x_slice: List[float], model: GPy.models.GPRegression, sample: bopt.Sample,
        show_marginal: int) -> Slice1D:
    param = experiment.hyperparameters[i]

    grid = param.range.grid(resolution)

    X_plot = np.zeros([resolution, n_dims], dtype=np.float32)

    for dim in range(n_dims):
        if dim == i:
            X_plot[:, dim] = grid
        else:
            X_plot[:, dim] = np.full([resolution], x_slice[dim], dtype=np.float32)

    X_plot_marginal = grid.reshape(-1, 1)

    if show_marginal == 1:
        others = experiment.predictive_samples_before(sample)
        X_m, Y_m = bopt.SampleCollection(others).to_xy()
        X_m = X_m[:, i].reshape(-1, 1)
        model = create_gp_for_data(experiment, [param], X_m, Y_m)

        mu, var = model.predict(X_plot_marginal)
    else:
        mu, var = model.predict(X_plot)

    mu = mu.reshape(-1)
    sigma = np.sqrt(var).reshape(-1)

    acq = bopt.ExpectedImprovement().raw_call(mu, sigma, model.Y.max())\
            .reshape(-1)

    other_samples: Dict[str, List[float]] = defaultdict(list)

    for other in experiment.predictive_samples_before(sample):
        other_x, other_y = other.to_xy()
        other_x = float(other_x.tolist()[i])

        if param.range.is_logscale():
            other_x = 10.0 ** other_x

        other_samples["x"].append(other_x)
        other_samples["y"].append(other_y)

    if param.range.is_logscale():
        x_slice_at = 10.0 ** x_slice[i]
        x = 10.0 ** grid
    else:
        x_slice_at = x_slice[i]
        x = grid

    return Slice1D(param, x.tolist(), x_slice_at, mu.tolist(),
                   sigma.tolist(), acq.tolist(), other_samples,
                   model)

def create_slice_2d(i: int, j: int, experiment: bopt.Experiment,
        resolution: int, n_dims: int, x_slice: List[float], model: GPy.models.GPRegression,
        sample: bopt.Sample, show_marginal: int) -> Slice2D:

    p1 = experiment.hyperparameters[i]
    p2 = experiment.hyperparameters[j]

    d1 = p1.range.grid(resolution)
    d2 = p2.range.grid(resolution)

    g1, g2 = np.meshgrid(d1, d2)

    gs = [0.0] * len(x_slice)
    gs[i] = g1
    gs[j] = g2

    for dim in range(len(x_slice)):
        if dim not in [i, j]:
            gs[dim] = np.full(g1.shape, x_slice[dim])

    grid = np.stack(gs, axis=-1)

    X_pred = grid.reshape(resolution * resolution, -1)

    if show_marginal == 1:
        others = experiment.predictive_samples_before(sample)
        X_m, Y_m = bopt.SampleCollection(others).to_xy()
        X_m = X_m[:, [i, j]].reshape(-1, 2)
        model = create_gp_for_data(experiment, [p1, p2], X_m, Y_m)

        mu, var = model.predict(X_pred[:, [i, j]])
    else:
        mu, var = model.predict(X_pred)

    mu = mu.reshape(-1)
    sigma = np.sqrt(var).reshape(-1)

    acq = bopt.ExpectedImprovement().raw_call(mu, sigma, model.Y.max())\
            .reshape(-1)

    mu    = mu.reshape(resolution, resolution)
    sigma = sigma.reshape(resolution, resolution)
    acq   = acq.reshape(resolution, resolution)

    other_samples: Dict[str, List[float]] = defaultdict(list)

    for other in experiment.predictive_samples_before(sample):
        other_x, other_y = other.to_xy()
        other_x1 = float(other_x.tolist()[i])
        other_x2 = float(other_x.tolist()[j])

        if p1.range.is_logscale():
            other_x1 = 10.0 ** other_x1

        if p2.range.is_logscale():
            other_x2 = 10.0 ** other_x2

        other_samples["x1"].append(other_x1)
        other_samples["x2"].append(other_x2)
        other_samples["y"].append(other_y)

    if p1.range.is_logscale():
        x1 = (10.0 ** d1).tolist()
        x1_slice_at = 10.0 ** x_slice[i]
    else:
        x1 = d1.tolist()
        x1_slice_at = x_slice[i]

    if p2.range.is_logscale():
        x2 = (10.0 ** d2).tolist()
        x2_slice_at = 10.0 ** x_slice[j]
    else:
        x2 = d2.tolist()
        x2_slice_at = x_slice[j]

    return Slice2D(p1, p2, x1, x2, x1_slice_at, x2_slice_at,
                   mu.tolist(), other_samples, model)


def run(args) -> None:
    import inspect
    import os
    script_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    app = Flask(__name__, template_folder=os.path.join(script_dir, "..",
        "templates"))
    app.debug = True

    app.config["port"] = args.port

    print("web path", app.root_path)

    @app.route("/")
    def index():
        with handle_cd_revertible(args), acquire_lock():
            experiment = bopt.Experiment.deserialize()
            experiment.collect_results()

            # TODO: zbytek asi neni potreba mit pod lockem, ale pak nejde
            #       cist output

            sample_results = [s.result for s in experiment.samples if s.result]
            sample_results_cummax = np.maximum.accumulate(sample_results).tolist()

            kernel_param_timeline = defaultdict(list)

            sorted_samples = sorted(experiment.samples, key=lambda x: x.created_at)

            num_random = len([s for s in sorted_samples if s.model.sampled_from_random_search()])

            for i, sample in enumerate(sorted_samples):
                if i < num_random + 1:
                    continue

                for key, value in sample.model.params.items():
                    if isinstance(value, list):
                        for v, h in zip(value, experiment.hyperparameters):
                            kernel_param_timeline["{}_{}".format(key, h.name)].append(v)
                    else:
                        kernel_param_timeline[key].append(value)

            n_dims = len(experiment.hyperparameters)

            sample_id = int(request.args.get("sample_id") or -1)
            show_acq = int(request.args.get("show_acq") or 0)
            show_marginal = int(request.args.get("show_marginal") or 1)

            sample = next((s for s in experiment.samples if s.job and s.job.job_id == sample_id), None)

            random_search_picked = False
            if sample and sample.model.sampled_from_random_search():
                random_search_picked = True

            print("picked sample", sample)

            slices_1d = []
            slices_2d = []
            resolution = 80

            if sample and not random_search_picked:
                x_slice = sample.hyperparam_values.x

                X_sample, Y_sample = experiment.get_xy()
                gpy_model = bopt.GPyModel.from_model_params(experiment.gp_config, sample.model, X_sample, Y_sample)

                model = gpy_model.model

                for i in range(n_dims):
                    for j in range(n_dims):
                        if i == j:
                            slices_1d.append(create_slice_1d(i, experiment,
                                resolution, n_dims, x_slice, model, sample, show_marginal))

                        elif i < j:
                            slices_2d.append(create_slice_2d(i, j, experiment,
                                resolution, n_dims, x_slice, model, sample, show_marginal))


            return render_template("index.html",
                    experiment=experiment,

                    sample_results=sample_results,
                    sample_results_cummax=sample_results_cummax,

                    kernel_param_timeline=kernel_param_timeline,

                    picked_sample=sample,

                    CollectFlag=bopt.CollectFlag,

                    slices_1d=slices_1d,
                    slices_2d=slices_2d,

                    sorted_samples=sorted_samples,

                    random_search_picked=random_search_picked,

                    show_acq=show_acq,
                    show_marginal=show_marginal,

                    sample_id=sample_id,
                    )


    server = Server(app.wsgi_app)
    server.watch("bopt/**/*")
    server.serve(host="0.0.0.0", port=app.config.get("port"))
