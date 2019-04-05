import logging
import numpy as np
from typing import Dict, List, Union

from bopt.basic_types import Hyperparameter, Discrete, ParamTypes, LogscaleFloat, LOGSCALE_BASE


class HyperparamValues:
    mapping: Dict[Hyperparameter, ParamTypes]
    x: np.ndarray

    def __init__(self, mapping: Dict[Hyperparameter, ParamTypes], x: np.ndarray) -> None:
        self.mapping = mapping
        self.x = x

    def to_dict(self) -> dict:
        return self.x.tolist()

    def __str__(self) -> str:
        # TODO: use this instead of other variants
        return str({h.name: (round(v, 2) if isinstance(v, float) else v)
                    for h, v in self.mapping.items()})

    def validate(self) -> bool:
        all_valid = True
        for param, value in self.mapping.items():
            param_valid = param.validate(value)

            if not param_valid:
                logging.error("Invalid hyperparam value {} for {}".format(value, param))

            all_valid = all_valid and param_valid

        if all_valid:
            logging.debug("All parameter values passed validation")

        return all_valid

    def similar_to(self, other: "HyperparamValues") -> bool:
        return all([param.range.compare_values(value, other.mapping[param])
                    for param, value in self.mapping.items()])

    @staticmethod
    def sample_params(hyperparameters: List[Hyperparameter]) -> np.ndarray:
        mapping = {h: h.range.sample() for h in hyperparameters}
        job_params = HyperparamValues.from_mapping(mapping)

        return job_params.x

    @staticmethod
    def mapping_from_vector(x: np.ndarray, hyperparameters:
            List[Hyperparameter]) -> "HyperparamValues":

        ########################
        # mapping: Dict[Hyperparameter, ParamTypes] = dict()
        #
        # for val, p in zip(x, hyperparameters):
        #     mapping[p] = p.range.inverse_map(val)
        #
        # return HyperparamValues(mapping, x)
        ######################


        # TODO: fuj, use map instead?

        # TODO: tady nastava problem, protoze log je aplikovany az potom, a tim ze
        #       se to rovnou zaokrouhli pak dostavam jenom 2^n
        typed_vals = [int(x) if p.range.is_discrete() else float(x)
                      for x, p in zip(x, hyperparameters)]

        # TODO: map back or just stick both in one struct?
        # TODO: unify naming
        # params_dict -> EvaluationArgs
        # ...
        mapping: Dict[Hyperparameter, ParamTypes] = dict(zip(hyperparameters,
            typed_vals))

        for p in hyperparameters:
            # TODO: properly check for map/inverse_map
            # TODO: rename map -> transform
            # if p.range.should_transform():
            mapping[p] = p.range.inverse_map(mapping[p])

        return HyperparamValues(mapping, x)

    # TODO: test! forward and back
    @staticmethod
    def from_mapping(mapping: Dict[Hyperparameter, ParamTypes]) -> np.ndarray:
        # TODO: 64 or 32 bit?
        x = np.zeros(len(mapping), dtype=np.float64)

        for i, key in enumerate(sorted(mapping, key=lambda k: k.name)):
            x[i] = key.range.map(mapping[key])

            # TODO: smazat fuj, uz neni potreba :)
            # value = mapping[key]
            #
            # if key.range.is_discrete():
            #     if isinstance(key.range, Discrete):
            #         x[i] = key.range.map(value)
            #     else:
            #         x[i] = int(value)
            # elif isinstance(key.range, LogscaleFloat):
            #     x[i] = key.range.map(value)
            # else:
            #     x[i] = float(value)

        return HyperparamValues(mapping, x)

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
