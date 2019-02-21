import logging
from bopt.models.model import Model
from bopt.models.random_search import RandomSearch
from bopt.models.gpy_model import GPyModel


class ModelLoader:
    @staticmethod
    def from_dict(data: dict) -> Model:
        if data is None:
            logging.info("ModelLoader received model = None.")
            return None

        model_type = data["model_type"]

        if model_type == "random_search":
            return RandomSearch()
        elif model_type == "gpy":
            return GPyModel.from_dict(data["gpy"])
