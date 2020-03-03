import logging

import numpy as np
from typing import List, Tuple

from bopt.hyperparam_values import HyperparamValues
from bopt.models.model import Model, SampleCollection
from bopt.basic_types import Hyperparameter, Bound
from bopt.models.parameters import ModelParameters


class RandomSearch(Model):
    @staticmethod
    def predict_next(hyperparameters: List[Hyperparameter]) -> HyperparamValues:
        logging.error("TODO: extract this and re-use across RoundingKernelWrapper, RS, GPm")
        mapping = {h: h.range.sample() for h in hyperparameters}

        for h, v in mapping.items():
            if isinstance(v, int) or isinstance(v, float):
                mapping[h] = h.maybe_round(np.array([v])).item()
            else:
                logging.warning("TODO IGNORING %s during round", h)

        return HyperparamValues.from_mapping(mapping)

    def to_model_params(self) -> ModelParameters:
        return ModelParameters("random_search", {}, "", "")


# TODO: delete
# def default_from_bounds(bounds: List[Bound]) -> np.ndarray:
#     x_0 = np.zeros(len(bounds))
#
#     for i, bound in enumerate(bounds):
#         if bound.type == "int":
#             x_0[i] = np.random.randint(bound.low, bound.high)
#         elif bound.type == "float":
#             x_0[i] = np.random.uniform(bound.low, bound.high)
#         else:
#             raise RuntimeError(f"Invalid type of bound, got {type(bound)}")
#
#     return x_0
#
