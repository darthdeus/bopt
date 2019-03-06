import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd

def run(args) -> None:
    handle_cd(args)

    if os.path.exists("meta.yml"):
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(args.meta_dir)

        run_params = vars(args).copy()
        del run_params["bopt"]
        del run_params["func"]
        del run_params["meta_dir"]

        experiment.manual_run(args.meta_dir, run_params, bopt.ModelParameters.manual_run())
    else:
        print(f"No meta.yml found.")
        sys.exit(1)
