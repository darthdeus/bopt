import os
import sys

import bopt
from bopt.cli.util import handle_cd_revertible, acquire_lock


def run(args) -> None:
    with handle_cd_revertible(args.dir):
        with acquire_lock():
            if os.path.exists("meta.yml"):
                print("Found existing meta.yml, resuming experiment.")

                experiment = bopt.Experiment.deserialize()
                experiment.collect_results()

                next_params, fitted_model = experiment.suggest()

                param_str = "\n".join([f"{key.name}: {value}"
                    for key, value in next_params.mapping.items()])

                param_args = " ".join([f"--{key.name}={value}"
                    for key, value in next_params.mapping.items()])

                print(f"""PARAMS:

{param_str}

To evaluate this manually, run:

bopt manual-run {param_args}
""")
            else:
                print("No meta.yml found.")
                sys.exit(1)
