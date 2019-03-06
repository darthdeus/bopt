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

        experiment.run_single(bopt.GPyModel(), args.meta_dir)
    else:
        print("No meta.yml found.")
        sys.exit(1)
