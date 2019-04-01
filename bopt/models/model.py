import abc
import os

from typing import List, Tuple
from bopt.basic_types import Hyperparameter
from bopt.hyperparam_values import HyperparamValues
from bopt.sample import SampleCollection
from bopt.models.parameters import ModelParameters


class Model(abc.ABC):
    # @abc.abstractmethod
    # def predict_next(self,
    #                  hyperparameters: List[Hyperparameter],
    #                  samples: "SampleCollection") -> Tuple[HyperparamValues, "Model"]:
    #     pass

    @abc.abstractmethod
    def to_model_params(self) -> ModelParameters:
        pass

