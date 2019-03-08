import numpy as np
from scipy.optimize import minimize

import GPy
from GPy.models import GPRegression

from typing import Tuple, List

import bopt.acquisition_functions.acquisition_functions as acq
from bopt.basic_types import Hyperparameter, Bound
from bopt.models.model import Model
from bopt.sample import Sample, SampleCollection
from bopt.models.parameters import ModelParameters
from bopt.run_params import RunParams


# TODO: split into multiple, serialization separate?
# TODO: round indexes
# https://arxiv.org/abs/1706.03673
class GPyModel(Model):
    model_name = "gpy"
    kernel_names = ["rbf", "Mat32", "Mat52"]
    acquisition_fn_names = ["ei", "pi"]

    model: GPRegression
    acquisition_fn: acq.AcquisitionFunction

    def __init__(self, model: GPRegression, acquisition_fn: acq.AcquisitionFunction) -> None:
        self.model = model
        self.acquisition_fn = acquisition_fn

    def to_model_params(self) -> ModelParameters:
        params = {
            name: float(self.model[name])
            for name in self.model.parameter_names()
        }

        return ModelParameters(
                GPyModel.model_name,
                params,
                self.model.kern.name,
                self.acquisition_fn.name())

    @staticmethod
    def from_model_params(model_params: ModelParameters, X, Y) -> "GPyModel":
        kernel_cls = GPyModel.parse_kernel_name(model_params.kernel)
        kernel = kernel_cls(input_dim=X.shape[1])

        model = GPRegression(X, Y, kernel=kernel)

        for name, value in model_params.params.items():
            model[name] = value

        acquisition_fn = GPyModel.parse_acquisition_fn(model_params.acquisition_fn)

        return GPyModel(model, acquisition_fn)

    @staticmethod
    def parse_kernel_name(name):
        if name == "rbf":
            return GPy.kern.RBF
        elif name == "Mat32":
            return GPy.kern.Matern32
        elif name == "Mat52":
            return GPy.kern.Matern52
        else:
            raise NotImplemented(f"Unknown kernel name '{name}'.")

    @staticmethod
    def parse_acquisition_fn(name):
        if name == "ei":
            return acq.ExpectedImprovement()
        elif name == "pi":
            return acq.ProbabilityOfImprovement()
        else:
            raise NotImplemented(f"Unknown acquisition function '{name}'.")

    def predict_next(self): raise NotImplemented("This should not be called, deprecated")

    def gpy_regression(run_params: RunParams,
            X_sample: np.ndarray, Y_sample: np.ndarray) -> GPRegression:
        # If there is only one sample, .std() == 0 and Y ends up being NaN.
        normalizer = len(X_sample) > 1

        # TODO: zkontrolovat, ze se kernely vyrabi jenom na jednom miste
        kernel = GPyModel.parse_kernel_name(run_params.kernel)(X_sample.shape[1])
        # TODO: nechybi normalizer i jinde?
        # TODO: predava se kernel a acq vsude?
        model = GPRegression(X_sample, Y_sample, kernel=kernel, normalizer=normalizer)

        # TODO: zamyslet se
        # model.kern.variance.set_prior(GPy.priors.Gamma(1., 0.1))
        # model.kern.lengthscale.set_prior(GPy.priors.Gamma(1., 0.1))
        model.kern.variance.unconstrain()
        model.kern.variance.constrain_bounded(1e-2, 1e6)

        model.kern.lengthscale.unconstrain()
        model.kern.lengthscale.constrain_bounded(1e-2, 1e6)

        model.Gaussian_noise.variance.unconstrain()
        model.Gaussian_noise.variance.constrain_bounded(1e-2, 1e6)

        # model.Gaussian_noise.set_prior(GPy.priors.Gamma(1., 0.1))
        model.optimize()

        return model

    @staticmethod
    def predict_next(run_params: RunParams, hyperparameters: List[Hyperparameter],
            X_sample: np.ndarray, Y_sample: np.ndarray) -> Tuple[dict, "Model"]:
        # TODO: compare NLL with and without normalizer

        model = GPyModel.gpy_regression(run_params, X_sample, Y_sample)
        acquisition_fn = GPyModel.parse_acquisition_fn(run_params.acquisition_fn)

        bounds = [b.range for b in hyperparameters]

        x_next = propose_location(acquisition_fn,
                model,
                Y_sample.reshape(-1).max(), # TODO: bez reshape?
                bounds)

        typed_vals = [int(x) if p.range.type == "int" else float(x)
                      for x, p in zip(x_next, hyperparameters)]

        names = [p.name for p in hyperparameters]

        params_dict = dict(zip(names, typed_vals))

        fitted_model = GPyModel(model, acquisition_fn)

        return params_dict, fitted_model


def propose_location(
    acquisition_fn: acq.AcquisitionFunction,
    gp: GPRegression,
    y_max: float,
    bounds: List[Bound],
    n_restarts: int = 25,
) -> np.ndarray:
    def min_obj(X):
        return -acquisition_fn(gp, X.reshape(1, -1), y_max)

    scipy_bounds = [(bound.low, bound.high) for bound in bounds]

    starting_points = []
    for _ in range(n_restarts):
        starting_points.append(np.array([bound.sample() for bound in bounds]))

    min_val = 1
    min_x = None

    for x0 in starting_points:
        res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method="L-BFGS-B")
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x
