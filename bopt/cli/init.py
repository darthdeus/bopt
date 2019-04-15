import yaml
import os
import sys
import inspect
import pathlib
import logging
from typing import Type, Callable, List

import bopt
from bopt import Hyperparameter
from bopt.cli.util import handle_cd, acquire_lock


def run(args) -> None:
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    handle_cd(args)

    with acquire_lock():
        hyperparameters: List[Hyperparameter] = []

        for param in args.param:
            name, type, *values = param.split(":")

            cls: Type
            parser: Callable

            if type == "discrete":
                hyp = bopt.Hyperparameter(name, bopt.Discrete(values))
            else:
                if type == "int":
                    cls = bopt.Integer
                    parser = int
                    low, high = values
                elif type == "float":
                    cls = bopt.Float
                    parser = float
                    low, high = values
                elif type == "logscale_float":
                    cls = bopt.LogscaleFloat
                    parser = float
                    low, high = values
                elif type == "logscale_int":
                    cls = bopt.LogscaleInt
                    parser = int
                    low, high = values
                elif type == "discrete":
                    continue
                else:
                    # TODO: pouzit reflexi, nehardcodit to
                    logging.error("Invalid value {} for hyperparameter type, "
                        "only 'int', 'float' and 'discrete' "
                        "are permitted.".format(type))
                    sys.exit(1)

                assert cls is not None

                hyp = bopt.Hyperparameter(name, cls(parser(low), parser(high)))

            hyperparameters.append(hyp)

        script_path = args.command
        default_arguments = args.arguments

        runner: bopt.Runner

        manual_arg_fnames: List[str] = args.manual_arg_fname

        if args.runner == "local":
            runner = bopt.LocalRunner(script_path, default_arguments,
                    manual_arg_fnames)
        elif args.runner == "sge":
            runner = bopt.SGERunner(script_path, default_arguments,
                    args.qsub or [], manual_arg_fnames)
        else:
            logging.error("Invalid value {} for runner,"
                "only 'local' and 'sge' are allowed.".format(args.runner))
            sys.exit(1)

        default_result_regex = "RESULT=(.*)"

        gp_config = bopt.GPConfig(args)

        experiment = bopt.Experiment(hyperparameters,
                runner, default_result_regex, gp_config)

        experiment.serialize()

        logging.info(f"Experiment created, run `bopt run -C {args.dir}` to start.")
