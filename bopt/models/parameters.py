class ModelParameters:
    model_name: str
    params: dict

    def __init__(self, model_name: str, params: dict) -> None:
        self.model_name = model_name
        self.params = params

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "params": self.params,
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelParameters":
        return ModelParameters(data["model_name"], data["params"])

    @staticmethod
    def manual_run() -> "ModelParameters":
        return ModelParameters("manual", {})
