class ModelParameters:
    model_name: str
    params: dict
    kernel: str

    def __init__(self, model_name: str, params: dict, kernel: str) -> None:
        self.model_name = model_name
        self.params = params
        self.kernel = kernel

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "params": self.params,
            "kernel": self.kernel
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelParameters":
        return ModelParameters(data["model_name"], data["params"], data["kernel"])

    @staticmethod
    def manual_run() -> "ModelParameters":
        return ModelParameters("manual", {}, None)
