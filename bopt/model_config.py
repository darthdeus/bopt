from argparse import Namespace

class ModelConfig:
    kernel: str
    acquisition_fn: str

    def __init__(self, args):
        self.kernel = args.kernel
        self.acquisition_fn = args.acquisition_fn

    @staticmethod
    def default() -> "ModelConfig":
        ns = Namespace()
        ns.kernel = "Mat52"
        ns.acquisition_fn = "ei"

        return ModelConfig(ns)

    def __str__(self) -> str:
        return "kernel: {}, acq_fn: {}".format(self.kernel,
                self.acquisition_fn)
