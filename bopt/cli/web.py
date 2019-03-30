import sys
import jsonpickle
import numpy as np

from typing import NamedTuple, List
from livereload import Server
from flask import Flask
from flask import render_template

import bopt
from bopt.cli.util import handle_cd, acquire_lock

class PosteriorSlice(NamedTuple):
    param: bopt.Hyperparameter
    x: List[float]
    y: List[float]
    std: List[float]
    points_x: List[float]
    points_y: List[float]
    gp: object # TODO: missing typing: bopt.GaussianProcessRegressor


def run(args) -> None:
    handle_cd(args)

    import inspect
    import os
    script_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    app = Flask(__name__, template_folder=os.path.join(script_dir, "..", "templates"))
    app.debug = True

    app.config["port"] = args.port

    print("web path", app.root_path)

    @app.route("/")
    def index():
        with acquire_lock():
            experiment = bopt.Experiment.deserialize()

        sample_results = [s.result for s in experiment.samples if s.result]
        sample_results_cummax = np.maximum.accumulate(sample_results).tolist()

        return render_template("index.html",
                experiment=experiment,
                sample_results=sample_results,
                sample_results_cummax=sample_results_cummax,
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
    server.watch("**/*")
    server.serve(port=app.config.get("port"))
