import jsonpickle
import numpy as np

import matplotlib.pyplot as plt

from typing import NamedTuple, List
from livereload import Server
from flask import Flask
from flask import render_template

import bopt

app = Flask(__name__)
app.debug = True


class PosteriorSlice(NamedTuple):
    param: bopt.Hyperparameter
    x: List[float]
    y: List[float]


@app.route("/")
def index():
    experiment = bopt.Experiment.deserialize("results/rl-monte-carlo")
    optim_result = experiment.current_optim_result()

    dimensions = []

    slices = []

    for i, param in enumerate(optim_result.params):
        dimensions.append({
            "values": optim_result.X_sample[:, i].tolist(),
            "range": [param.range.low, param.range.high],
            "label": param.name,
        })

        x, y = optim_result.slice_at(i)

        slices.append(PosteriorSlice(param, x.tolist(), y.tolist()))

    mu_mat, extent, gx, gy = bopt.plot_2d_optim_result(optim_result)
    exp_gp = bopt.base64_plot()

    heatmap = []
    for i in range(len(mu_mat)):
        heatmap.append(mu_mat[i, :].tolist())

    minval = min(np.min(heatmap), np.min(optim_result.y_sample))
    maxval = max(np.max(heatmap), np.max(optim_result.y_sample))

    data = {
        "posterior_slices": slices,
        "best_x": optim_result.best_x.tolist(),
        "best_y": optim_result.best_y,
        "minval": minval,
        "maxval": maxval,
        "colors": optim_result.y_sample.tolist(),
        "dimensions": dimensions,
        "heatmap": {
            "z": heatmap,
            "x": gx.tolist(),
            "y": gy.tolist(),
            "sx": optim_result.X_sample[:, 0].tolist(),
            "sy": optim_result.X_sample[:, 1].tolist(),
            "sz": optim_result.y_sample.tolist(),
        }
    }
    json_data = jsonpickle.dumps(data)

    return render_template("index.html", data=data, json_data=json_data, experiment=experiment)


server = Server(app.wsgi_app)
server.watch("**/*")
server.serve()
