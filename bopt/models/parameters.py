# TODO: difference between this and ModelConfig?
class ModelParameters:
    model_name: str
    # TODO: type
    params: dict
    kernel: str
    acquisition_fn: str

    def __init__(self, model_name: str, params: dict, kernel: str, acquisition_fn: str) -> None:
        self.model_name = model_name
        self.params = params
        self.kernel = kernel
        self.acquisition_fn = acquisition_fn

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "params": self.params,
            "kernel": self.kernel,
            "acquisition_fn": self.acquisition_fn
        }

    def sampled_from_random_search(self) -> bool:
        return self.model_name == "random_search"

    # TODO: fuj tohle uz nechci?
    def can_predict_mean(self) -> bool:
        # This is simply to avoid cyclic imports
        from bopt.models.gpy_model import GPyModel

        return self.model_name == GPyModel.model_name

    def __str__(self) -> str:
        param_str = " ".join("{}={}".format(n, (round(v, 2) if not isinstance(v, list) else list(map(lambda x: round(x, 2), v)))) for n, v in self.params.items())
        return "k:{}, acq:{}, p:{}".format(
                self.kernel,
                self.acquisition_fn,
                param_str)

    @staticmethod
    def from_dict(data: dict) -> "ModelParameters":
        return ModelParameters(
                data["model_name"],
                data["params"],
                data["kernel"],
                data["acquisition_fn"])

    @staticmethod
    def for_manual_run() -> "ModelParameters":
        return ModelParameters("manual", {}, "", "")
