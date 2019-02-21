import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize

from bopt.acquisition_functions.acquisition_functions import AcquisitionFunction, expected_improvement
from bopt.models.model import Model, SampleCollection
from bopt.basic_types import Hyperparameter, Bound
from bopt.models.gaussian_process_regressor import GaussianProcessRegressor


# TODO: BOModel?
class GPModel(Model):
    gp: GaussianProcessRegressor

    def to_serializable(self) -> "Model":
        model = GPModel()
        model.gp = self.gp.to_serializable()
        return model

    def from_serializable(self) -> "Model":
        model = GPModel()
        model.gp = self.gp.from_serializable()
        return model
        pass

    def predict_next(self, hyperparameters: List[Hyperparameter],
                     sample_col: SampleCollection) -> Tuple[dict, "Model"]:
        X_sample, y_sample = sample_col.to_xy()

        gp = GaussianProcessRegressor().fit(X_sample, y_sample).optimize_kernel()

        bounds = [b.range for b in hyperparameters]

        x_next = propose_location(expected_improvement,
                gp,
                y_sample.max(),
                bounds)

        typed_vals = [int(x) if p.range.type == "int" else float(x)
                      for x, p in zip(x_next, hyperparameters)]

        names = [p.name for p in hyperparameters]

        params_dict = dict(zip(names, typed_vals))

        fitted_model = GPModel()
        fitted_model.gp = gp

        return params_dict, fitted_model


def propose_location(
    acquisition: AcquisitionFunction,
    gp: GaussianProcessRegressor,
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
