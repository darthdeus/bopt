import numpy as np
from typing import Dict, List, Union

from bopt.basic_types import Hyperparameter, Discrete


ParamTypes = Union[float, int, str]


class JobParams:
    mapping: Dict[Hyperparameter, ParamTypes]
    x: np.ndarray

    def __init__(self, mapping: Dict[Hyperparameter, ParamTypes], x: np.ndarray) -> None:
        self.mapping = mapping
        self.x = x

    def to_dict(self) -> dict:
        # TODO: chybi, naimplementovat!
        raise NotImplementedError()

    @staticmethod
    def mapping_from_vector(x: np.ndarray, hyperparameters: List[Hyperparameter]) \
            -> "JobParams":

        typed_vals = [int(x) if p.range.is_discrete() else float(x)
                      for x, p in zip(x, hyperparameters)]

        # TODO: map back or just stick both in one struct?
        # TODO: unify naming
        # params_dict -> EvaluationArgs
        # ...
        mapping: Dict[Hyperparameter, ParamTypes] = dict(zip(hyperparameters, typed_vals))

        for p in hyperparameters:
            if isinstance(p.range, Discrete):
                mapping[p] = p.range.inverse_map(mapping[p])

        return JobParams(mapping, x)


    # TODO: test! forward and back
    @staticmethod
    def from_mapping(mapping: Dict[Hyperparameter, ParamTypes]) -> np.ndarray:
        x = np.zeros(len(mapping), dtype=np.float64)

        for i, key in enumerate(sorted(mapping, key=lambda k: k.name)):
            value = mapping[key]

            if key.range.is_discrete():
                if isinstance(key.range, Discrete):
                    x[i] = key.range.map(value)
                else:
                    x[i] = int(value)
            else:
                x[i] = float(value)

        return x


    # @property
    # def x(self):
    #     return self.param_dict_to_x()
    #
    # def param_dict_to_x(self) -> np.ndarray:
    #     x = np.zeros(len(self.mapping), dtype=np.float64)
    #
    #     for i, key in enumerate(sorted(self.mapping.keys(), key=lambda k: k.name)):
    #         value = self.mapping[key]
    #
    #         x[i] = value
    #
    #     return x
