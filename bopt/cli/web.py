import math
import sys
import jsonpickle
import numpy as np

from typing import NamedTuple, List
from livereload import Server
from flask import Flask, render_template, request

import bopt
from bopt.cli.util import handle_cd_revertible, acquire_lock

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

        noise_values = []
        ls_values = []
        sigma_values = []

        kernel_params_list = [
            ("noise", noise_values),
            ("lengthscale", ls_values),
            ("sigma", sigma_values)
        ]

        for sample in sorted(experiment.samples, key=lambda x: x.created_at):
            if sample.model.sampled_from_random_search():
                continue

            p = sample.model.params

            noise_values.append(p["Gaussian_noise.variance"])
            ls_values.append(p["rbf.lengthscale"])
            sigma_values.append(math.sqrt(p["rbf.variance"]))

        n_dims = len(experiment.hyperparameters)

        sample_id = request.args.get("sample_id") or -1

        sample = next((s for s in experiment.samples if s.job and s.job.job_id == int(sample_id)), None)

        # sample = experiment.samples[-1]

        print("picked sample", sample)

        diagonal_x = []
        diagonal_mu = []
        diagonal_mu_bounds = []
        diagonal_sigma = []

        if sample:
            x_slice = sample.hyperparam_values.x

            X_sample, Y_sample = experiment.get_xy()
            gpy_model = bopt.GPyModel.from_model_params(sample.model, X_sample, Y_sample)

            model = gpy_model.model

            for i in range(n_dims):
                for j in range(n_dims):
                    if i == j:

                        param = experiment.hyperparameters[i]

                        resolution = 50

                        # if isinstance(param.range, bopt.LogscaleInt) or isinstance(param.range, bopt.LogscaleFloat):
                            # TODO: logscale
                            # grid = np.linspace(param.range.low, param.range.high, num=resolution)
                        # else:
                        #     grid = np.linspace(param.range.low, param.range.high, num=resolution)

                        grid = np.linspace(param.range.low, param.range.high, num=resolution)

                        X_plot = np.zeros([resolution, n_dims], dtype=np.float32)

                        for dim in range(n_dims):
                            if dim == i:
                                X_plot[:, dim] = np.full([resolution], x_slice[dim], dtype=np.float32)
                            else:
                                X_plot[:, dim] = grid

                        mu, var = model.predict(X_plot)
                        mu = mu.reshape(-1)
                        sigma = np.sqrt(var).reshape(-1)


                        diagonal_x.append(grid.tolist())
                        diagonal_mu.append(mu.tolist())
                        diagonal_mu_bounds.append([min(mu), max(mu)])
                        diagonal_sigma.append(sigma.tolist())

        return render_template("index.html",
                experiment=experiment,
                sample_results=sample_results,
                sample_results_cummax=sample_results_cummax,
                kernel_params_list=kernel_params_list,

                picked_sample=sample,

                CollectFlag=bopt.CollectFlag,

                diagonal_x=diagonal_x,
                diagonal_mu=diagonal_mu,
                diagonal_mu_bounds=diagonal_mu_bounds,
                diagonal_sigma=diagonal_sigma)


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
