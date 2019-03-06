import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml

def run(args) -> None:
    handle_cd(args)

    with ensure_meta_yml():
        experiment = bopt.Experiment.deserialize(".")

        run_params = vars(args).copy()
        del run_params["bopt"]
        del run_params["func"]
        if "dir" in run_params:
            del run_params["dir"]

        experiment.manual_run(".", run_params, bopt.ModelParameters.manual_run())
