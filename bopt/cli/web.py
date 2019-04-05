from collections import defaultdict
import math
import sys
import jsonpickle
import numpy as np

from typing import NamedTuple, List, Tuple, Dict
from livereload import Server
from flask import Flask, render_template, request

import bopt
from bopt.cli.util import handle_cd_revertible, acquire_lock

class Slice1D:
    param: bopt.Hyperparameter
    x: List[float]

    x_slice_at: float

    mu: List[float]
    sigma: List[float]
    acq: List[float]

    other_samples: Dict[str, List[float]]

    def __init__(self, param: bopt.Hyperparameter,
            x: List[float],
            x_slice_at: float,
            mu: List[float],
            sigma: List[float],
            acq: List[float],
            other_samples: Dict[str, List[float]]) -> None:
        self.param = param
        self.x = x
        self.x_slice_at = x_slice_at
        self.mu = mu
        self.sigma = sigma
        self.acq = acq
        self.other_samples = other_samples

    def sigma_low(self) -> List[float]:
        return [m - s for m, s in zip(self.mu, self.sigma)]

    def sigma_high(self) -> List[float]:
        return [m + s for m, s in zip(self.mu, self.sigma)]

    def mu_bounds(self) -> Tuple[float, float]:
        other_results = self.other_samples["y"]

        return min(self.sigma_low() + self.acq + other_results), \
               max(self.sigma_high() + self.acq + other_results)

    def x_range(self) -> Tuple[float, float]:
        low = self.param.range.low
        high = self.param.range.high

        if self.param.range.is_logscale():
            low = math.log10(low)
            high = math.log10(high)

        margin = (high - low) * 0.05

        return low - margin, high + margin


class PosteriorSlice(NamedTuple):
    param: bopt.Hyperparameter
    x: List[float]
    y: List[float]
    std: List[float]
    points_x: List[float]
    points_y: List[float]
    gp: object # TODO: missing typing: bopt.GaussianProcessRegressor


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

        sample_results = [s.result for s in experiment.samples if s.result]
        sample_results_cummax = np.maximum.accumulate(sample_results).tolist()

        kernel_param_timeline = defaultdict(list)

        sorted_samples = sorted(experiment.samples, key=lambda x: x.created_at)

        num_random = len([s for s in sorted_samples if s.model.sampled_from_random_search()])

        for i, sample in enumerate(sorted_samples):
            if i < num_random + 1:
                continue

            for key, value in sample.model.params.items():
                kernel_param_timeline[key].append(value)

            # TODO: chci tohle?
            # if i < 9:
            #     continue

        n_dims = len(experiment.hyperparameters)

        sample_id = request.args.get("sample_id") or -1

        sample = next((s for s in experiment.samples if s.job and s.job.job_id == int(sample_id)), None)

        random_search_picked = False
        if sample and sample.model.sampled_from_random_search():
            random_search_picked = True

        # sample = experiment.samples[-1]

        print("picked sample", sample)

        slices_1d = []

        # diagonal_x = []
        #
        # diagonal_mu = []
        # diagonal_mu_bounds = []
        #
        # diagonal_sigma = []
        # diagonal_sigma_low = []
        # diagonal_sigma_high = []
        #
        # diagonal_acq = []
        # diagonal_acq_bounds = []

        picked_sample_x = None

        if sample and not random_search_picked:
            # picked_sample_x = sample.hyperparam_values.rescaled_values_for_plot(experiment.hyperparameters)
            picked_sample_x = np.round(sample.hyperparam_values.x, 2)

            x_slice = sample.hyperparam_values.x

            X_sample, Y_sample = experiment.get_xy()
            gpy_model = bopt.GPyModel.from_model_params(sample.model, X_sample, Y_sample)

            model = gpy_model.model

            for i in range(n_dims):
                for j in range(n_dims):
                    if i == j:
                        param = experiment.hyperparameters[i]

                        resolution = 80

                        # if isinstance(param.range, bopt.LogscaleInt) or isinstance(param.range, bopt.LogscaleFloat):
                            # TODO: logscale
                            # grid = np.linspace(param.range.low, param.range.high, num=resolution)
                        # else:
                        #     grid = np.linspace(param.range.low, param.range.high, num=resolution)

                        if param.range.is_logscale():
                            grid = np.linspace(math.log10(param.range.low), math.log10(param.range.high), num=resolution)
                        else:
                            grid = np.linspace(param.range.low, param.range.high, num=resolution)

                        X_plot = np.zeros([resolution, n_dims], dtype=np.float32)

                        for dim in range(n_dims):
                            if dim == i:
                                X_plot[:, dim] = grid
                            else:
                                X_plot[:, dim] = np.full([resolution], x_slice[dim], dtype=np.float32)

                        mu, var = model.predict(X_plot)
                        mu = mu.reshape(-1)
                        sigma = np.sqrt(var).reshape(-1)

                        acq = bopt.ExpectedImprovement().raw_call(mu, sigma, model.Y.max())\
                                .reshape(-1)

                        # TODO: je tohle spravne?

                        x_slice_at = picked_sample_x[i]

                        if param.range.is_logscale():
                            x_slice_at = 10.0 ** picked_sample_x[i]
                            x_plot = 10.0 ** grid
                        else:
                            x_plot = grid

                        other_samples = {
                            "x": [],
                            "y": [],
                        }

                        for other in experiment.samples:
                            if other.created_at < sample.created_at:
                                other_x, other_y = other.to_xy()
                                other_x = float(other_x.tolist()[i])

                                if param.range.is_logscale():
                                    other_x = 10.0 ** other_x

                                other_samples["x"].append(other_x)
                                other_samples["y"].append(other_y)

                        slice1d = Slice1D(param,
                                x_plot.tolist(),
                                x_slice_at,
                                mu.tolist(),
                                sigma.tolist(),
                                acq.tolist(),
                                other_samples)

                        slices_1d.append(slice1d)


        return render_template("index.html",
                experiment=experiment,

                sample_results=sample_results,
                sample_results_cummax=sample_results_cummax,

                kernel_param_timeline=kernel_param_timeline,

                picked_sample=sample,
                picked_sample_x=picked_sample_x,

                CollectFlag=bopt.CollectFlag,

                slices_1d=slices_1d,

                sorted_samples=sorted_samples,

                random_search_picked=random_search_picked,

                # diagonal_x=diagonal_x,
                #
                # diagonal_mu=diagonal_mu,
                # diagonal_mu_bounds=diagonal_mu_bounds,
                #
                # diagonal_sigma=diagonal_sigma,
                # diagonal_sigma_low=diagonal_sigma_low,
                # diagonal_sigma_high=diagonal_sigma_high,
                #
                # diagonal_acq=diagonal_acq,
                # diagonal_acq_bounds=diagonal_acq_bounds,
                )


    # @app.route("/")
    # def index():
    #     experiment = bopt.Experiment.deserialize()
    #     optim_result = experiment.current_optim_result()
    #
    #     sample_col = bopt.SampleCollection(experiment.samples)
    #
    #     gp = optim_result.fit_gp()
    #
    #     dimensions = []
    #
    #     slices = []
    #
    #     for i, param in enumerate(optim_result.params):
    #         dimensions.append({
    #             "values": optim_result.X_sample[:, i].tolist(),
    #             "range": [param.range.low, param.range.high],
    #             "label": param.name,
    #         })
    #
    #         x, y, std = optim_result.slice_at(i, gp)
    #
    #         points_x = optim_result.X_sample[:, i].tolist()
    #         points_y = optim_result.y_sample.tolist()
    #
    #         posterior_slice = PosteriorSlice(
    #             param,
    #             x.tolist(),
    #             y.tolist(),
    #             std.tolist(),
    #             points_x,
    #             points_y,
    #             gp
    #         )
    #
    #         slices.append(posterior_slice)
    #
    #     mu_mat, extent, gx, gy = bopt.plot_2d_optim_result(optim_result, gp=gp)
    #
    #     heatmap = []
    #     for i in range(len(mu_mat)):
    #         heatmap.append(mu_mat[i, :].tolist())
    #
    #     minval = min(np.min(heatmap).item(), np.min(optim_result.y_sample).item())
    #     maxval = max(np.max(heatmap).item(), np.max(optim_result.y_sample).item())
    #
    #     minval = min(optim_result.y_sample)
    #     maxval = max(optim_result.y_sample)
    #
    #     y_range = maxval - minval
    #
    #     minval -= y_range * 0.2
    #     maxval += y_range * 0.2
    #
    #     data = {
    #         "posterior_slices": slices,
    #         "best_x": optim_result.best_x.tolist(),
    #         "best_y": optim_result.best_y,
    #         "minval": minval,
    #         "maxval": maxval,
    #         "colors": optim_result.y_sample.tolist(),
    #         "dimensions": dimensions,
    #         "heatmap": {
    #             "z": heatmap,
    #             "x": gx.tolist(),
    #             "y": gy.tolist(),
    #             "sx": optim_result.X_sample[:, 0].tolist(),
    #             "sy": optim_result.X_sample[:, 1].tolist(),
    #             "sz": optim_result.y_sample.tolist(),
    #         }
    #     }
    #     json_data = jsonpickle.dumps(data)
    #
    #     param_traces = bopt.kernel_opt.get_param_traces()
    #
    #     nll_trace = param_traces["nll"]
    #     param_traces.pop("nll")
    #
    #     return render_template("index.html", data=data,
    #             json_data=json_data,
    #             experiment=experiment,
    #             sample_col=sample_col,
    #             param_traces=param_traces,
    #             nll_trace=nll_trace,
    #             result_gp=gp)


    server = Server(app.wsgi_app)
    server.watch("bopt/**/*")
    server.serve(host="0.0.0.0", port=app.config.get("port"))
