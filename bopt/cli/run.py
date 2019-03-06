import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml

def run(args) -> None:
    handle_cd(args)

    with ensure_meta_yml():
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(".")

        experiment.run_loop(bopt.models.gpy_model.GPyModel(), ".", n_iter=args.n_iter)
