import numpy as np
from typing import List, Tuple

from bopt.models.model import Model, SampleCollection
from bopt.basic_types import Hyperparameter, Bound


class RandomSearch(Model):
    def predict_next(self, hyperparameters: List[Hyperparameter],
                     samples: SampleCollection) -> Tuple[dict, "Model"]:

        value = {h.name: h.range.sample() for h in hyperparameters}

        return value, self

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
