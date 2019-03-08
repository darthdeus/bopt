class ModelParameters:
    model_name: str
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

    def can_predict_mean(self) -> bool:
        # This is simply to avoid cyclic imports
        # TODO: maybe use a global constant config that registers/lists predictive models?
        from bopt.models.gpy_model import GPyModel

        return self.model_name == GPyModel.model_name

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
