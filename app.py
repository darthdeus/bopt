import json
import numpy as np

import matplotlib.pyplot as plt

from livereload import Server
from flask import Flask
from flask import render_template

import bopt
from bopt import Experiment

app = Flask(__name__)
app.debug = True

def he():
    plt.figure()
    bounds = [bopt.Hyperparameter("a", bopt.Float(-1.0, 2.0))]
    noise = 0.2

    def f(X):
        return -np.sin(3*X) - X**2 + 0.7*X

    X_init = np.array([[-0.9], [1.1]])
    y_init = f(X_init)

    X_true = np.arange(bounds[0].range.low, bounds[0].range.high, 0.01).reshape(-1, 1)

    y_true = f(X_true)
    import random
    noisy_f = lambda x: f(x).item() + random.random()*0.01

    plt.figure()
    bopt.bo_plot_exploration(noisy_f, bounds, X_true=X_true, y_true=y_true,
            n_iter=4, gp_noise=0.02)

    return bopt.plot.base64_plot()

def experiment_gp(experiment: Experiment) -> str:
    X_train = np.array([list(e.run_parameters.values()) for e in experiment.evaluations])
    y_train = np.array([e.final_result() for e in experiment.evaluations])

    plt.figure()
    bopt.GaussianProcess().fit(X_train, y_train).plot_prior(np.linspace(0, 1))

    return bopt.base64_plot()


@app.route("/")
def index():
    experiment = Experiment.deserialize("results/rl-monte-carlo")
    x = np.arange(0, 2*np.pi, step=0.01)
    y = np.sin(x)
    image = np.random.randn(100, 100)

    optim_result = experiment.current_optim_result()

    dimensions = []

    for i, param in enumerate(optim_result.params):
        dimensions.append({
            "values": optim_result.X_sample[:, i].tolist(),
            "range": [param.range.low, param.range.high],
            "label": param.name,
        })

    mu_mat, extent, gx, gy = bopt.plot_2d_optim_result(optim_result)
    exp_gp = bopt.base64_plot()

    heatmap = []
    for i in range(len(mu_mat)):
        heatmap.append(mu_mat[i, :].tolist())

    data = {
        "experiment_gp": exp_gp,
    }

    json_data = json.dumps({
        "x": x.tolist(),
        "y": y.tolist(),
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
    })

    return render_template("index.html", data=data, json_data=json_data, experiment=experiment)


server = Server(app.wsgi_app)
server.watch("**/*")
server.serve()
