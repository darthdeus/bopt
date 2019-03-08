class RunParams:
    kernel: str
    acquisition_fn: str

    def __init__(self, args):
        self.kernel = args.kernel
        self.acquisition_fn = args.acquisition_fn
