import os
import sys
import yaml

import bopt


def run_loop(exp: bopt.Experiment) -> None:
    loop = bopt.OptimizationLoop(exp.hyperparameters)
    loop.run(exp, n_iter=20)


def run(args) -> None:
    meta_fname = os.path.join(
        args.DIR,
        "meta.yml"
    )

    config_fname = os.path.join(
        args.DIR,
        "config.yml"
    )

    if os.path.exists(meta_fname):
        print("Found existing meta.yml, resuming experiment.")
        experiment = bopt.Experiment.deserialize(args.DIR)

        run_loop(experiment)
        sys.exit(0)

    if not os.path.exists(config_fname):
        print(f"There is no `config.yml` at {config_fname}.")
        sys.exit(1)

    with open(config_fname, "rt") as f:
        config = yaml.load(f.read())

    meta_dir = args.DIR
    runner_config = config["runner"]

    if not runner_config["result_parser"] == "bopt.LastLineLastWordParser":
        print("Only `bopt.LastLineLastWordParser` is supported, got {}.".format(runner_config["result_parser"]))
        sys.exit(1)

    if runner_config["type"] == "local":
        runner = bopt.LocalRunner(
            meta_dir,
            runner_config["script"],
            runner_config["args"],
            bopt.LastLineLastWordParser()
        )
    else:
        # TODO: ...
        print("Only local runner is currently supported via `bopt run`, got {}.".format(runner_config["type"]))
        sys.exit(1)

    hyperparameters = []

    for param in config["hyperparameters"]:
        if param["type"] == "float":
            cls = bopt.Float
            parser = float
        elif param["type"] == "int":
            cls = bopt.Integer
            parser = int
        else:
            print("Invalid value {} for hyperparameter type, only 'int' and 'float' are permitted.".format(param["type"]))
            sys.exit(1)

        # TODO: low/high vs min/max? :(

        hyperparameters.append(
            bopt.Hyperparameter(
                param["name"],
                cls(
                    parser(param["bounds"]["min"]),
                    parser(param["bounds"]["max"])
                )
            )
        )

    experiment = bopt.Experiment(
        meta_dir,
        hyperparameters,
        runner)

    run_loop(experiment)
