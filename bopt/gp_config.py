from argparse import Namespace


class GPConfig:
    kernel: str
    acquisition_fn: str

    ard: bool
    fit_mean: bool
    gamma_prior: bool
    gamma_a: float
    gamma_b: float

    num_optimize_restarts: int

    def __init__(self, args):
        self.kernel = args.kernel
        self.acquisition_fn = args.acquisition_fn

        self.ard = args.ard == 1
        self.fit_mean = args.fit_mean == 1
        self.gamma_prior = args.gamma_prior == 1
        self.gamma_a = args.gamma_a
        self.gamma_b = args.gamma_b

        self.num_optimize_restarts = args.num_optimize_restarts

    def __str__(self) -> str:
        # TODO: pridat co chybi
        return "kernel: {}, acq_fn: {}".format(self.kernel,
                self.acquisition_fn)
