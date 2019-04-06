import os
import logging
import sys
import yaml
from tqdm import tqdm

from typing import List

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml, acquire_lock
from bopt.plot import plot_current

# TODO: co kdyz dostanu manual evaluation, zkusit precejenom fitnout model
#       ale do plotu napsat, ze ten model neni podle ceho byl vybrany?
def run(args) -> None:
    from bopt.models.gpy_model import GPyModel

    handle_cd(args)

    with acquire_lock(), ensure_meta_yml():
        experiment = bopt.Experiment.deserialize()
        experiment.collect_results()

        processed_samples: List[bopt.Sample] = []

        for sample in tqdm(experiment.samples_for_prediction()):
            if sample.model.model_name == GPyModel.model_name:
                sample_col = bopt.SampleCollection(processed_samples)
                X, Y = sample_col.to_xy()

                model = GPyModel.from_model_params(experiment.gp_config,
                        sample.model, X, Y)

                try:
                    plot_current(experiment, model, sample.to_x())
                except ValueError as e:
                    logging.error("Plotting failed {}".format(e))

            processed_samples.append(sample)
