from typing import Callable, Any, TypeVar, Generic, Optional, NamedTuple, Type, List
from argparse import Namespace


T = TypeVar("T")


class GPParam(Generic[T]):
    name: str
    type: Type[T]
    default: T
    action: Optional[str]
    help: str

    def __init__(self, name: str, type: Type[T], default: T, action: Optional[str], help: str):
        self.name = name
        self.type = type
        self.default = default
        self.action = action
        self.help = help


# TODO fill this
kernel_names = "..."
acq_fn_names = "..."


config_params: List[GPParam[Any]] = [
    GPParam[str]("kernel", str, "Mat52", None, f"Specifies the GP kernel. Allowed values are: {kernel_names}"),
    GPParam[str]("acquisition_fn", str, "ei", None,
                 f"Specifies the acquisition function. Allowed values are: {acq_fn_names}"),
    GPParam[int]("ard", int, 1, None, "Toggles automatic relevance determination (one lengthscale per hyperparameter)"),
    GPParam[int]("fit-mean", int, 1, None,
                 "When enabled the mean function is fit during kernel optimization. "
                 "Otherwise it is set to zero."),

    GPParam[int]("gamma-prior", int, 1, None,
                 "When enabled, kernel parameters will use a Gamma prior "
                 "instead of a hard constraint."),
    GPParam[float]("gamma-a", float, 1.0, None, "The shape parameter of the Gamma prior."),
    GPParam[float]("gamma-b", float, 0.1, None, "The inverse rate parameter of the Gamma prior."),

    GPParam[int]("informative-prior", int, 1, None,
                 "When enabled, kernel parameters use an informative Gamma prior on lengthscale."),

    GPParam[float]("acq-xi", float, 0.001, None, "The xi parameter of the acquisition functions."),
    GPParam[int]("acq-n-restarts", int, 25, None, "Number of restarts when optimizing the acquisition function."),
    GPParam[int]("num-optimize-restarts", int, 10, None, "Number of restarts during kernel optimization."),

    GPParam[bool]("random-search-only", bool, False, "store_true", "Only use random search when picking new hyperparameters."),
         # , required=False)

]


# TODO: generate CLI from this or generate this from CLI
class GPConfig:
    kernel: str
    acquisition_fn: str

    ard: bool
    fit_mean: bool
    gamma_prior: bool
    gamma_a: float
    gamma_b: float
    informative_prior: bool

    acq_xi: float
    acq_n_restarts: int

    num_optimize_restarts: int

    random_search_only: bool

    def __init__(self, args):
        self.kernel = args.kernel
        self.acquisition_fn = args.acquisition_fn

        self.ard = args.ard == 1
        self.fit_mean = args.fit_mean == 1

        self.gamma_prior = args.gamma_prior == 1
        self.gamma_a = args.gamma_a
        self.gamma_b = args.gamma_b
        self.informative_prior = args.informative_prior == 1

        self.acq_xi = args.acq_xi
        self.acq_n_restarts = args.acq_n_restarts

        self.num_optimize_restarts = args.num_optimize_restarts

        self.random_search_only = args.random_search_only

    def __str__(self) -> str:
        # TODO: pridat co chybi
        return "kernel: {}, acq_fn: {}".format(self.kernel, self.acquisition_fn)

    def to_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "acquisition_fn": self.acquisition_fn,
            "ard": self.ard,
            "fit_mean": self.fit_mean,
            "gamma_prior": self.gamma_prior,
            "gamma_a": self.gamma_a,
            "gamma_b": self.gamma_b,
            "informative_prior": self.informative_prior,
            "acq_xi": self.acq_xi,
            "acq_n_restarts": self.acq_n_restarts,
            "num_optimize_restarts": self.num_optimize_restarts,
            "random_search_only": self.random_search_only,
        }

    @staticmethod
    def from_dict(data: dict) -> "GPConfig":
        return GPConfig(
            data["kernel"],
            data["acquisition_fn"],
            data["ard"],
            data["fit_mean"],
            data["gamma_prior"],
            data["gamma_a"],
            data["gamma_b"],
            data["informative_prior"],
            data["acq_xi"],
            data["acq_n_restarts"],
            data["num_optimize_restarts"],
            data["random_search_only"],
        )
