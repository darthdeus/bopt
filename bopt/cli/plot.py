import os
import sys
import yaml
from tqdm import tqdm

from typing import List

import bopt
from bopt.cli.util import handle_cd

# TODO: co kdyz dostanu manual evaluation, zkusit precejenom fitnout model
#       ale do plotu napsat, ze ten model neni podle ceho byl vybrany?
def run(args) -> None:
    handle_cd(args)

    if os.path.exists("meta.yml"):
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(".")

        processed_samples: List[bopt.Sample] = []

        for sample in tqdm(experiment.samples):
            if sample.model.model_name == "gpy":
                sample_col = bopt.SampleCollection(processed_samples, ".")
                X, Y = sample_col.to_xy()

                model = bopt.models.gpy_model.GPyModel.from_model_params(sample.model, X, Y)

                experiment.plot_current(model, ".", sample.to_x())

            processed_samples.append(sample)
    else:
        print("No meta.yml found.")
        sys.exit(1)
