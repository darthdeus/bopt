import yaml
import os
import sys
import inspect
import bopt
from typing import Type, Callable, List
from bopt import Hyperparameter


def run(args) -> None:
    hyperparameters: List[Hyperparameter] = []

    for param in args.param:
        name, type, low, high = param.split(":")

        cls: Type
        parser: Callable

        if type == "int":
            cls = bopt.Integer
            parser = int
        elif type == "float":
            cls = bopt.Float
            parser = float
        else:
            print("Invalid value {} for hyperparameter type, only 'int' and 'float' are permitted.".format(type))
            sys.exit(1)

        assert cls is not None

        hyperparameters.append(
            bopt.Hyperparameter(name, cls(parser(low), parser(high)))
        )

    assert args.result_parser == "bopt.LastLineLastWordParser"
    # TODO: sge
    assert args.runner == "local"

    meta_dir = args.dir
    script_path = args.command
    default_arguments = args.arguments

    runner = bopt.LocalRunner(script_path, default_arguments)
    experiment = bopt.Experiment(hyperparameters, runner)

    import pathlib

    pathlib.Path(meta_dir).mkdir(parents=True, exist_ok=True)

    experiment.serialize(meta_dir)

    print(f"Experiment created, run `bopt run {meta_dir}` to start.")

    return
