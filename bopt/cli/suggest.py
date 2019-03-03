import os
import sys
import yaml
from tqdm import tqdm

from typing import List

import bopt

def run(args) -> None:
    meta_fname = os.path.join(
        args.meta_dir,
        "meta.yml"
    )

    if os.path.exists(meta_fname):
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(args.meta_dir)

        next_params, fitted_model = \
                experiment.suggest(bopt.models.gpy_model.GPyModel(), args.meta_dir)

        param_str = "\n".join([f"{key}: {value}" for key, value in next_params.items()])

        print("""PARAMS:

{}""".format(param_str))
    else:
        print(f"There is no `meta.yml` at {meta_fname}.")
        sys.exit(1)
