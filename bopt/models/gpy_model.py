import numpy as np
from scipy.optimize import minimize

import GPy
from GPy.models import GPRegression

from typing import Tuple, List

from bopt.acquisition_functions.acquisition_functions import AcquisitionFunction, expected_improvement
from bopt.basic_types import Hyperparameter, Bound
from bopt.models.model import Model, Sample, SampleCollection


# TODO: round indexes
# https://arxiv.org/abs/1706.03673
class GPyModel(Model):
    model: GPRegression

    def __init__(self, model: GPRegression = None) -> None:
        self.model = model

    def to_dict(self) -> dict:
        return {
            "model_type": "gpy",
            "gpy": self.model[:].tolist()
            # {
            #     "kernel": self.model.kern.name,
            #     "input_dim": self.model.kern.input_dim,
            #     "params": self.model.param_array.tolist(),
            #     "X": self.model.X, # TODO: tolist?
            #     "Y": self.model.Y,
            # }
        }

    @staticmethod
    def from_dict(data: dict) -> Model:
        # TODO: fuj naming
        gpy_model = GPyModel()

        # if data["kernel"] == "rbf":
        #     kernel = GPy.kern.RBF(input_dim=data["input_dim"])
        # else:
        #     raise NotImplemented()

        gpy_model# .model = GPRegression(data["X"], data["Y"], kernel)

        # gp = GPRegression.from_dict(data)
        # gpy_model.model = GPRegression.from_gp(gp)
        return gpy_model

    def predict_next(self, hyperparameters: List[Hyperparameter],
                     sample_col: SampleCollection) -> Tuple[dict, "Model"]:
        X_sample, y_sample = sample_col.to_xy()

        model = GPRegression(X_sample, y_sample.reshape(-1, 1))
        model.optimize()
        # gp = GaussianProcessRegressor().fit(X_sample, y_sample).optimize_kernel()

        bounds = [b.range for b in hyperparameters]

        x_next = propose_location(expected_improvement,
                model,
                y_sample.max(),
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
