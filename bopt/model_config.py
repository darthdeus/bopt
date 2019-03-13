from argparse import Namespace

class ModelConfig:
    kernel: str
    acquisition_fn: str

    def __init__(self, args):
        self.kernel = args.kernel
        self.acquisition_fn = args.acquisition_fn

    @staticmethod
    def default() -> "RunParams":
        ns = Namespace()
        ns.kernel = "Mat52"
        ns.acquisition_fn = "ei"

        return ModelConfig(ns)
