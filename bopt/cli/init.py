import os
import sys
import pathlib
import logging
import shutil
from typing import Type, Callable, List

import bopt
from bopt import Hyperparameter
from bopt.cli.util import handle_cd_revertible, acquire_lock


def run(args) -> None:
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(args.dir, "output")).mkdir(parents=True, exist_ok=True)

    with handle_cd_revertible(args.dir):
        with acquire_lock():
            hyperparameters: List[Hyperparameter] = []

            # TODO: nesmaze se lockfile kdyz to spadne

            for param in args.param:
                name, type, *values = param.split(":")

                cls: Type
                parser: Callable

                if type == "discrete":
                    hyp = bopt.Hyperparameter(name, bopt.Discrete(values))
                else:
                    mapping = {
                        "int": [bopt.Integer, int],
                        "float": [bopt.Float, float],
                        "logscale_int": [bopt.LogscaleInt, int],
                        "logscale_float": [bopt.LogscaleFloat, float],
                    }

                    if type in mapping:
                        cls, parser = mapping[type]
                    else:
                        logging.error("Invalid value {} for hyperparameter type, "
                                      "only 'int', 'float', 'logscale_int', 'logscale_float' "
                                      "and 'discrete' are permitted.".format(type))
                        sys.exit(1)

                    if len(values) == 2:
                        low, high = values
                        buckets = -1
                    elif len(values) == 3:
                        low, high, buckets = values
                    else:
                        logging.error("Invalid number of values in '%s', must be 2 or 3.", values)
                        sys.exit(1)

                    # assert issubclass(cls, bopt.Bound), "Expected bopt.Bound, got {}".format(cls)
                    hyp = bopt.Hyperparameter(name, cls(parser(low), parser(high), buckets))

                hyperparameters.append(hyp)

            script_path = args.command
            default_arguments = args.arguments

            runner: bopt.Runner

            manual_arg_fnames: List[str] = []

            for fname in args.manual_arg_fname:
                base_fname = os.path.basename(fname)
                shutil.copy(fname, "./{}".format(base_fname))

                manual_arg_fnames.append(base_fname)

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

            experiment = bopt.Experiment(args.task, args.batch, hyperparameters,
                                         runner, default_result_regex, gp_config)

            experiment.serialize()

            logging.info(f"Experiment created, run `bopt run -C {args.dir}` to start.")
