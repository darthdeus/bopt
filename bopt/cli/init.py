import yaml
import os
import sys
import inspect
import pathlib
from typing import Type, Callable, List

import bopt
from bopt import Hyperparameter


def run(args) -> None:
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
            elif type == "discrete":
                continue
            else:
                print("Invalid value {} for hyperparameter type, only 'int', 'float' and 'discrete' are permitted.".format(type))
                sys.exit(1)

            assert cls is not None

            hyp = bopt.Hyperparameter(name, cls(parser(low), parser(high)))

        hyperparameters.append(hyp)

    assert args.result_parser == "bopt.LastLineLastWordParser"

    meta_dir = args.dir
    script_path = args.command
    default_arguments = args.arguments

    if args.runner == "local":
        runner = bopt.LocalRunner(script_path, default_arguments)
    elif args.runner == "sge":
        runner = bopt.SGERunner(script_path, default_arguments)
    else:
        print("Invalid value {} for runner, only 'local' and 'sge' are allowed.".format(args.runner))

    default_result_regex = "RESULT=(.*)"
    experiment = bopt.Experiment(hyperparameters, runner, default_result_regex)

    pathlib.Path(meta_dir).mkdir(parents=True, exist_ok=True)

    experiment.serialize(meta_dir)

    print(f"Experiment created, run `bopt run {meta_dir}` to start.")

    return
