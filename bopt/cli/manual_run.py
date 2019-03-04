import os
import sys
import yaml

import bopt

def run(args) -> None:
    meta_fname = os.path.join(
        args.meta_dir,
        "meta.yml"
    )

    if os.path.exists(meta_fname):
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(args.meta_dir)

        run_params = vars(args).copy()
        del run_params["bopt"]
        del run_params["func"]
        del run_params["meta_dir"]

        experiment.manual_run(args.meta_dir, run_params, bopt.ModelParameters.manual_run())
    else:
        print(f"There is no `meta.yml` at {meta_fname}.")
        sys.exit(1)
