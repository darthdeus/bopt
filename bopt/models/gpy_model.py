import logging
import numpy as np
from scipy.optimize import minimize

import GPy
from GPy.models import GPRegression

from typing import Tuple, List

import bopt.acquisition_functions.acquisition_functions as acq
from bopt.basic_types import Hyperparameter, Bound, Discrete, OptimizationFailed
from bopt.models.model import Model
from bopt.sample import Sample
from bopt.models.parameters import ModelParameters
from bopt.gp_config import GPConfig
from bopt.hyperparam_values import HyperparamValues


# TODO: split into multiple, serialization separate?
# TODO: round indexes
# https://arxiv.org/abs/1706.03673
class GPyModel(Model):
    model_name = "gpy"

    model: GPRegression
    acquisition_fn: acq.AcquisitionFunction

    def __init__(self, model: GPRegression, acquisition_fn: acq.AcquisitionFunction) -> None:
        self.model = model
        self.acquisition_fn = acquisition_fn

    def to_model_params(self) -> ModelParameters:
        params = {
            name: (self.model[name].tolist() if self.model[name].size > 1 else float(self.model[name]))
            for name in self.model.parameter_names()
        }

        return ModelParameters(
                GPyModel.model_name,
                params,
                self.model.kern.name,
                self.acquisition_fn.name())

    @staticmethod
    def from_model_params(gp_config: GPConfig, model_params: ModelParameters, X, Y) -> "GPyModel":
        # TODO: check that these are actually GPy params
        kernel_cls = GPyModel.parse_kernel_name(model_params.kernel)

        # TODO: vyrabet kernel jenom na jednom miste
        kernel = kernel_cls(input_dim=X.shape[1], ARD=gp_config.ard)

        model = GPRegression(X, Y, kernel=kernel, normalizer=len(X) > 1)

        for name, value in model_params.params.items():
            model[name] = value

        acquisition_fn = GPyModel.parse_acquisition_fn(model_params.acquisition_fn)

        return GPyModel(model, acquisition_fn)

    @staticmethod
    def gpy_regression(hyperparameters: List[Hyperparameter],
            gp_config: GPConfig, X_sample: np.ndarray, Y_sample: np.ndarray) -> GPRegression:

        # TODO: zkontrolovat, ze se kernely vyrabi jenom na jednom miste
        kernel = GPyModel.parse_kernel_name(gp_config.kernel)(X_sample.shape[1], ARD=gp_config.ard)
        # TODO: predava se kernel a acq vsude?

        # If there is only one sample, .std() == 0 and Y ends up being NaN.
        model = GPRegression(X_sample, Y_sample, kernel=kernel, normalizer=len(X_sample) > 1)

        logging.debug("GPY hyperparam optimization start")

        min_bound = 1e-2
        max_bound = 1e3

        if gp_config.informative_prior:
            for i, param in enumerate(hyperparameters):
                prior = GPyModel.prior_for_hyperparam(gp_config, param)
                model.kern.lengthscale[[i]].set_prior(prior)

            variance_prior = GPy.priors.Gamma(gp_config.gamma_a, gp_config.gamma_b)

            model.kern.variance.unconstrain()
            model.kern.variance.set_prior(variance_prior)
        else:
            model.Gaussian_noise.variance.unconstrain()
            model.Gaussian_noise.variance.constrain_bounded(min_bound, max_bound)

            if gp_config.gamma_prior:
                model.kern.variance.set_prior(GPy.priors.Gamma(gp_config.gamma_a, gp_config.gamma_b))
                model.kern.lengthscale.set_prior(GPy.priors.Gamma(gp_config.gamma_a, gp_config.gamma_b))
            else:
                model.kern.variance.unconstrain()
                model.kern.variance.constrain_bounded(min_bound, max_bound)

                model.kern.lengthscale.unconstrain()
                model.kern.lengthscale.constrain_bounded(min_bound, max_bound)

        model.optimize_restarts(gp_config.num_optimize_restarts)

        logging.debug("GPY hyperparam optimization DONE, params: {}".format(model.param_array))

        return model

    @staticmethod
    def predict_next(gp_config: GPConfig,
            hyperparameters: List[Hyperparameter],
            X_sample: np.ndarray, Y_sample: np.ndarray) -> Tuple[HyperparamValues, "Model"]:
        # TODO: compare NLL with and without normalizer
        assert not np.any(np.isnan(Y_sample))

        model = GPyModel.gpy_regression(hyperparameters, gp_config, X_sample, Y_sample)
        acquisition_fn = GPyModel.parse_acquisition_fn(gp_config.acquisition_fn)

        x_next = GPyModel.propose_location(acquisition_fn, model, Y_sample.max(),
                hyperparameters, gp_config)

        job_params = HyperparamValues.mapping_from_vector(x_next, hyperparameters)

        fitted_model = GPyModel(model, acquisition_fn)

        return job_params, fitted_model

    @staticmethod
    def propose_location(acquisition_fn: acq.AcquisitionFunction, gp:
            GPRegression, y_max: float, hyperparameters: List[Hyperparameter],
            gp_config: GPConfig) -> np.ndarray:

        def min_obj(X):
            return -acquisition_fn(gp, X.reshape(1, -1), y_max, gp_config.acq_xi)

        scipy_bounds = [h.range.scipy_bound_tuple() for h in
                hyperparameters]

        starting_points = []
        for _ in range(gp_config.acq_n_restarts):
            starting_points.append(HyperparamValues.sample_params(hyperparameters))

        min_val = 1e9
        min_x = None

        logging.debug("Starting propose_location")

        for i, x0 in enumerate(starting_points):
            res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method="L-BFGS-B")

            if np.any(np.isnan(res.fun[0])):
                logging.error("Ran into NAN during {}/{} acq fn optimization, got {}".format(i, len(starting_points), res.fun))

            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        if min_x is None:
            logging.error("Optimization failed {}-times with GP params {}".format(len(starting_points), gp.param_array))
            raise OptimizationFailed(gp.param_array)

        logging.debug("Finished propose_location")

        return min_x

    @staticmethod
    def parse_kernel_name(name):
        if name == "rbf":
            return GPy.kern.RBF
        elif name == "Mat32":
            return GPy.kern.Matern32
        elif name == "Mat52":
            return GPy.kern.Matern52
        else:
            raise NotImplementedError(f"Unknown kernel name '{name}'.")

    @staticmethod
    def parse_acquisition_fn(name):
        if name == "ei":
            return acq.ExpectedImprovement()
        elif name == "pi":
            return acq.ProbabilityOfImprovement()
        else:
            raise NotImplementedError(f"Unknown acquisition function '{name}'.")

    @staticmethod
    def prior_for_hyperparam(gp_config: GPConfig, param: Hyperparameter) -> GPy.priors.Prior:
        d = param.range.high - param.range.low
        mid = (d / 2.0) + param.range.low

        if d > 1:
            return GPy.priors.Gamma.from_EV(mid, (d/4.0)**2.0)
        else:
            return GPy.priors.Gamma(gp_config.gamma_a, gp_config.gamma_b)


