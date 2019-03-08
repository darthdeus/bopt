import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml
from bopt.run_params import RunParams


def run(args) -> None:
    handle_cd(args)

    with ensure_meta_yml():
        experiment = bopt.Experiment.deserialize(".")

        # TODO: unify naming run_params vs model_params
        model_params = vars(args).copy()
        del model_params["bopt"]
        del model_params["func"]
        if "dir" in model_params:
            del model_params["dir"]

        experiment.manual_run(RunParams.default(), ".", model_params,
                bopt.ModelParameters.for_manual_run())
