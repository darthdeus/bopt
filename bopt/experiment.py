import yaml
import os
import psutil
import time
import pathlib
import numpy as np

from typing import List, Optional, Tuple

from bopt.models.model import Model
from bopt.sample import Sample, SampleCollection
from bopt.models.model_loader import ModelLoader
from bopt.models.parameters import ModelParameters
from bopt.models.random_search import RandomSearch
from bopt.basic_types import Hyperparameter
from bopt.runner.abstract import Job, Runner
from bopt.runner.runner_loader import RunnerLoader

from bopt.optimization_result import OptimizationResult


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

    def run_next(self, model: Model, meta_dir: str, output_dir: str) -> Tuple[Job, Model]:
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

        return job, fitted_model

    def run_loop(self, model: Model, meta_dir: str, n_iter=10) -> None:
        output_dir = pathlib.Path(meta_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_iter):
            job, fitted_model = self.run_next(model, meta_dir, str(output_dir))

            while not job.is_finished():
                psutil.wait_procs(psutil.Process().children(), timeout=0.01)
                time.sleep(1)

            self.serialize(meta_dir)

            ##################################
            # TODO: plot with GPy!!!
            ##################################
            self.plot_current(fitted_model, meta_dir)

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


    def plot_current(self, model: Model, meta_dir: str, resolution: float = 30, noise: float = 0.0):
        # TODO: handle more than 2 dimensions properly
        # assert len(result.params) == 2

        b1 = self.hyperparameters[0].range
        b2 = self.hyperparameters[1].range

        # TODO: float64
        x1 = np.linspace(b1.low, b1.high, resolution, dtype=np.float64)
        x2 = np.linspace(b2.low, b2.high, resolution, dtype=np.float64)

        assert len(x1) < 80, f"too large x1, len = {len(x1)}"
        assert len(x2) < 80, f"too large x1, len = {len(x2)}"

        gx, gy = np.meshgrid(x1, x2)

        X_2d = np.c_[gx.ravel(), gy.ravel()]

        bounds = [p.range for p in self.hyperparameters]

        X_sample, y_sample = SampleCollection(self.samples, meta_dir).to_xy()
        X_sample = X_sample[:, :2]

        if isinstance(model, RandomSearch):
            return

        assert model is not None, "gp is None"

        # TODO: with_bounds?
        # gp = GaussianProcess(kernel=result.kernel.with_bounds(bounds)) \
        #     .fit(X_sample, result.y_sample) \
        #     .optimize_kernel()

        # TODO: lol :)
        model = model.model

        mu, std = model.predict(X_2d)
        # mu, _ = gp.mu_std()

        # TODO: use paramz properly
        param_str = "ls: {:.3f}, variance: {:.3f}, noise: {:.3f}".format(
                float(model.kern.lengthscale),
                float(model.kern.variance),
                float(model.Gaussian_noise.variance)

        )
        # param_str = str(gp.kern) + " " + str(gp.noise) + f" nll={round(gp.log_prob().numpy().item(), 2)}"

        mu_mat = mu.reshape(gx.shape[0], gx.shape[1])
        extent = [b1.low, b1.high, b2.high, b2.low]

        import matplotlib.pyplot as plt
        import datetime
        # assert result.best_x is not None

        U_LB = os.environ.get("USE_LBFGS", False)
        U_TF = os.environ.get("USE_TF", False)

        # plt.title(f"LBFGS={U_LB} TF={U_TF}   noise={round(gp.noise, 2)} {result.kernel}", fontsize=20)
        # plt.pcolor(mu_mat, extent=extent, aspect="auto")
        plt.title(param_str)
        plt.pcolor(gx, gy, mu_mat, cmap="jet")
        plt.scatter(X_sample[:, 0], X_sample[:, 1], c="k")
        # plt.scatter([result.best_x[0]], [result.best_x[1]], c="r")
        plt.savefig("tmp/opt-plot-{}.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))

        return mu_mat, extent, x1, x2
