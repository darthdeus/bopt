import numpy as np
from scipy.optimize import minimize

import GPy
from GPy.models import GPRegression

from typing import Tuple, List

from bopt.acquisition_functions.acquisition_functions import AcquisitionFunction, expected_improvement
from bopt.basic_types import Hyperparameter, Bound
from bopt.models.model import Model
from bopt.sample import Sample, SampleCollection
from bopt.models.parameters import ModelParameters


# TODO: round indexes
# https://arxiv.org/abs/1706.03673
class GPyModel(Model):
    model: GPRegression

    def __init__(self, model: GPRegression = None) -> None:
        self.model = model

    def to_model_params(self) -> ModelParameters:
        # TODO: kernel
        params = {
            name: float(self.model[name])
            for name in self.model.parameter_names()
        }

        return ModelParameters("gpy", params, self.model.kern.name)

    @staticmethod
    def from_model_params(model_params: ModelParameters, X, Y) -> "GPyModel":
        kernel_cls = GPyModel.parse_kernel_name(model_params.kernel)
        kernel = kernel_cls(input_dim=X.shape[1])

        model = GPRegression(X, Y, kernel=kernel)

        for name, value in model_params.params.items():
            model[name] = value

        return GPyModel(model)

    @staticmethod
    def parse_kernel_name(name):
        if name == "rbf":
            return GPy.kern.RBF
        elif name == "Mat32":
            return GPy.kern.Matern32
        elif name == "Mat52":
            return GPy.kern.Matern52
        else:
            raise NotImplemented(f"Unknown kernel name {name}.")

    def to_dict(self) -> dict:
        pass
    #     return {
    #         "model_type": "gpy",
    #         "gpy": {name: float(self.model[name])
    #                 for name in self.model.parameter_names()}
    #         # {
    #         #     "kernel": self.model.kern.name,
    #         #     "input_dim": self.model.kern.input_dim,
    #         #     "params": self.model.param_array.tolist(),
    #         #     "X": self.model.X, # TODO: tolist?
    #         #     "Y": self.model.Y,
    #         # }
    #     }
    #
    # @staticmethod
    # def from_dict(data: dict) -> Model:
    #     # TODO: fuj naming
    #     gpy_model = GPyModel()
    #
    #     # if data["kernel"] == "rbf":
    #     #     kernel = GPy.kern.RBF(input_dim=data["input_dim"])
    #     # else:
    #     #     raise NotImplemented()
    #
    #     gpy_model# .model = GPRegression(data["X"], data["Y"], kernel)
    #
    #     # gp = GPRegression.from_dict(data)
    #     # gpy_model.model = GPRegression.from_gp(gp)
    #     return gpy_model

    def predict_next(self, hyperparameters: List[Hyperparameter],
                     sample_col: SampleCollection) -> Tuple[dict, "Model"]:
        X_sample, Y_sample = sample_col.to_xy()

        # TODO: compare NLL with and without normalizer

        # If there is only one sample, .std() == 0 and Y ends up being NaN.
        normalizer = len(X_sample) > 1

        model = GPRegression(X_sample, Y_sample, normalizer=normalizer)

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

        bounds = [b.range for b in hyperparameters]

        x_next = propose_location(expected_improvement,
                model,
                Y_sample.reshape(-1).max(), # TODO: bez reshape?
                bounds)

        typed_vals = [int(x) if p.range.type == "int" else float(x)
                      for x, p in zip(x_next, hyperparameters)]

        names = [p.name for p in hyperparameters]

        params_dict = dict(zip(names, typed_vals))

        fitted_model = GPyModel()
        fitted_model.model = model

        return params_dict, fitted_model


def propose_location(
    acquisition: AcquisitionFunction,
    gp: GPRegression,
    y_max: float,
    bounds: List[Bound],
    n_restarts: int = 25,
) -> np.ndarray:
    def min_obj(X):
        return -acquisition(gp, X.reshape(1, -1), y_max)

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
