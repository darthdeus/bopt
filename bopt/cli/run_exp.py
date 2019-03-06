import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd

def run(args) -> None:
    handle_cd(args)

    if os.path.exists("meta.yml"):
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(args.DIR)

        # TODO
        experiment.run_loop(bopt.models.gpy_model.GPyModel(), args.DIR, n_iter=args.n_iter)
    else:
        print("No meta.yml found.")
        sys.exit(1)
