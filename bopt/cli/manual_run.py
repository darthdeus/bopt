import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml, acquire_lock


def run(args) -> None:
    handle_cd(args)

    with acquire_lock(), ensure_meta_yml():
        experiment = bopt.Experiment.deserialize()
        experiment.collect_results()

        # TODO: unify naming run_params vs model_params
        model_params = vars(args).copy()
        del model_params["bopt"]
        del model_params["func"]
        if "dir" in model_params:
            del model_params["dir"]

        mapping = {}

        for hyperparam in experiment.hyperparameters:
            if hyperparam.name in model_params:
                mapping[hyperparam] = \
                    hyperparam.range.parse(model_params[hyperparam.name])
            else:
                print("\nMissing value for: {}".format(hyperparam.name))
                sys.exit(1)

        hyperparam_values = bopt.HyperparamValues.from_mapping(mapping)

        if not hyperparam_values.validate():
            print("Some values of hyperparameters were invalid, exiting.")
            sys.exit(1)

        next_sample, _ = experiment.manual_run(hyperparam_values,
                bopt.ModelParameters.for_manual_run())

        print("Started {}".format(next_sample))
