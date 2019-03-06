import os
import sys
import yaml
from tqdm import tqdm

from typing import List

import bopt
from bopt.cli.util import handle_cd

def run(args) -> None:
    handle_cd(args)

    if os.path.exists("meta.yml"):
        print("Found existing meta.yml, resuming experiment.")

        experiment = bopt.Experiment.deserialize(".")

        next_params, fitted_model = \
                experiment.suggest(bopt.models.gpy_model.GPyModel(), ".")

        param_str = "\n".join([f"{key}: {value}" for key, value in next_params.items()])
        param_args = " ".join([f"--{key}={value}" for key, value in next_params.items()])

        print(f"""PARAMS:

{param_str}

To evaluate this manually, run:

bopt manual-run {"."} {param_args}""")
    else:
        print("No meta.yml found.")
        sys.exit(1)
