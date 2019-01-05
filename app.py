import json
import numpy as np

import myopt
import matplotlib.pyplot as plt

from livereload import Server
from flask import Flask
from flask import render_template

from myopt import Experiment

app = Flask(__name__)
app.debug = True

def experiment_gp(experiment: Experiment) -> str:
    X_train = np.array([list(e.run_parameters.values()) for e in experiment.evaluations])
    y_train = np.array([e.final_result() for e in experiment.evaluations])

    myopt.GaussianProcess().fit(X_train, y_train).plot_prior(np.linspace(0, 1))

    return myopt.plot.base64_plot()


@app.route("/")
def index():
    experiment = Experiment.deserialize("results/rl-monte-carlo")
    x = np.arange(0, 2*np.pi, step=0.01)
    y = np.sin(x)
    image = np.random.randn(100, 100)

    myopt.bayesian_optimization.plot_2d_optim_result(experiment.current_optim_result())
    exp_gp = myopt.plot.base64_plot()

    data = {
        "experiment_gp": exp_gp
    }

    json_data = json.dumps({
        "x": x.tolist(),
        "y": y.tolist()
    })

    return render_template("index.html", data=data, json_data=json_data, experiment=experiment)

server = Server(app.wsgi_app)
server.watch("**/*")
server.serve()
