import yaml
import os
import sys
import inspect
import bopt
from typing import Type, Callable


def run(args) -> None:
    hyperparameters = []

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

    print(f"Experiment created, run `bopt run {meta_dir}` to start.")

    return

    script_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    yaml_template_fname = os.path.join(
        script_dir,
        "..",
        "templates",
        "new_experiment.yml"
    )

    with open(yaml_template_fname, "r") as f:
        yaml_template = f.read()

    name = os.path.basename(args.DIR)

    yaml_template = yaml_template.replace("EXPERIMENT_NAME", name)

    if not os.path.exists(args.DIR):
        print(f"Directory {args.DIR} doesn't exist, creating.")
        os.mkdir(args.DIR)

    target_fname = os.path.join(args.DIR, "config.yml")

    with open(target_fname, "wt") as f:
        f.write(yaml_template)

    print(f"Created a new experiment at {args.DIR}")
