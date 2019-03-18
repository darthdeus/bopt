import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml, acquire_lock
from bopt.models.gpy_model import GPyModel

def run(args) -> None:
    handle_cd(args)

    with acquire_lock(), ensure_meta_yml():
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(".")

        experiment.run_loop(bopt.ModelConfig(args), ".", n_iter=args.n_iter)
