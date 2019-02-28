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

        for sample in experiment.samples:
            if sample.model.model_name == "gpy":
                model = bopt.GPyModel.from_model_params(sample.model, experiment)
                __import__('ipdb').set_trace()
    else:
        print(f"There is no `meta.yml` at {meta_fname}.")
        sys.exit(1)
