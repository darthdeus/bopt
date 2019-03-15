import logging
import numpy as np
from scipy.optimize import minimize

import GPy
from GPy.models import GPRegression

from typing import Tuple, List

import bopt.acquisition_functions.acquisition_functions as acq
from bopt.basic_types import Hyperparameter, Bound, Discrete
from bopt.models.model import Model
from bopt.sample import Sample, SampleCollection
from bopt.models.parameters import ModelParameters
from bopt.model_config import ModelConfig
from bopt.job_params import JobParams


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

        model = GPRegression(X, Y, kernel=kernel, normalizer=len(X) > 1)

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

    def gpy_regression(model_config: ModelConfig,
            X_sample: np.ndarray, Y_sample: np.ndarray) -> GPRegression:

        # TODO: zkontrolovat, ze se kernely vyrabi jenom na jednom miste
        kernel = GPyModel.parse_kernel_name(model_config.kernel)(X_sample.shape[1])
        # TODO: predava se kernel a acq vsude?

        # If there is only one sample, .std() == 0 and Y ends up being NaN.
        model = GPRegression(X_sample, Y_sample, kernel=kernel, normalizer=len(X_sample) > 1)

        # TODO: zamyslet se
        # model.kern.variance.set_prior(GPy.priors.Gamma(1., 0.1))
        # model.kern.lengthscale.set_prior(GPy.priors.Gamma(1., 0.1))

        min_bound = 1e-2
        max_bound = 1e3

        logging.info("GPY hyperparam optimization start")

        model.kern.variance.unconstrain()
        model.kern.variance.constrain_bounded(min_bound, max_bound)

        model.kern.lengthscale.unconstrain()
        model.kern.lengthscale.constrain_bounded(min_bound, max_bound)

        model.Gaussian_noise.variance.unconstrain()
        model.Gaussian_noise.variance.constrain_bounded(min_bound, max_bound)

        # model.Gaussian_noise.set_prior(GPy.priors.Gamma(1., 0.1))
        model.optimize()

        logging.info("GPY hyperparam optimization DONE, params: {}".format(model.param_array))

        return model

    @staticmethod
    def predict_next(model_config: ModelConfig, hyperparameters: List[Hyperparameter],
            X_sample: np.ndarray, Y_sample: np.ndarray) -> Tuple[JobParams, "Model"]:
        # TODO: compare NLL with and without normalizer

        model = GPyModel.gpy_regression(model_config, X_sample, Y_sample)
        acquisition_fn = GPyModel.parse_acquisition_fn(model_config.acquisition_fn)

        x_next = GPyModel.propose_location(acquisition_fn, model, Y_sample.max(),
                hyperparameters)

        logging.info("New proposed location at x = {}".format(x_next))

        job_params = JobParams.mapping_from_vector(x_next, hyperparameters)

        fitted_model = GPyModel(model, acquisition_fn)

        return job_params, fitted_model

    @staticmethod
    def propose_location( acquisition_fn: acq.AcquisitionFunction, gp:
            GPRegression, y_max: float, hyperparameters: List[Hyperparameter],
            n_restarts: int = 25,) -> np.ndarray:
        # TODO: heh
        # np.seterrcall(lambda *args: __import__('ipdb').set_trace())
        np.seterr(all="warn")

        def min_obj(X):
            return -acquisition_fn(gp, X.reshape(1, -1), y_max)

        scipy_bounds = [(h.range.low, h.range.high) for h in hyperparameters]

        starting_points = []
        for _ in range(n_restarts):
            # TODO: tohle spadne protoze sample z discrete takhle nejde pouzit
            x_sample = JobParams.sample_params(hyperparameters)

            # starting_points.append(np.array([bound.sample() for bound in
            # bounds]))
            starting_points.append(x_sample)

        min_val = 1e9
        min_x = None

        logging.info("Starting propose_location")

        for x0 in starting_points:
            res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method="L-BFGS-B")

            assert not np.any(np.isnan(res.fun))

            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        assert min_x is not None

        new_point_str = " ".join(map(lambda xx: str(round(xx, 2)), min_x.tolist()))
        logging.info("Finished optimizing acq_fn, got a new min at {}".format(new_point_str))

        return min_x
