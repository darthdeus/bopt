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

    @staticmethod
    def from_dict(data: dict) -> "ModelParameters":
        return ModelParameters(
                data["model_name"],
                data["params"],
                data["kernel"],
                data["acquisition_fn"])

    @staticmethod
    def manual_run() -> "ModelParameters":
        return ModelParameters("manual", {}, "", "")
