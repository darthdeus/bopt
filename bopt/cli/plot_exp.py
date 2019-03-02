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

        processed_samples: List[bopt.Sample] = []

        for sample in tqdm(experiment.samples):
            if sample.model.model_name == "gpy":
                sample_col = bopt.SampleCollection(processed_samples, args.meta_dir)
                X, Y = sample_col.to_xy()

                model = bopt.models.gpy_model.GPyModel.from_model_params(sample.model, X, Y)

                experiment.plot_current(model, args.meta_dir, sample.to_x())

            processed_samples.append(sample)
    else:
        print(f"There is no `meta.yml` at {meta_fname}.")
        sys.exit(1)
