from flask import Flask
from flask import render_template
import json
import numpy as np

from myopt import Experiment

app = Flask(__name__)


@app.route("/")
def hello():
    experiment = Experiment.deserialize("results/meta-dir")
    x = np.arange(0, 2*np.pi, step=0.01)
    y = np.sin(x)

    data = json.dumps({
        "x": x.tolist(),
        "y": y.tolist()
        })

    return render_template("hello.html", data=data, experiment=experiment)

