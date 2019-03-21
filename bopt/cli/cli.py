import sys
import os
import psutil
import inspect
import argparse

from bopt.models.gpy_model import GPyModel
from bopt.cli import (
    cli,
    exp,
    init,
    jobstat,
    manual_run,
    plot,
    run,
    run_single,
    suggest,
    util,
    web,
    debug,
    clean
)


def main():
    parser = argparse.ArgumentParser(prog="bopt")
    parser.set_defaults()

    cd_parser = argparse.ArgumentParser(add_help=None)
    cd_parser.add_argument(
        "-C",
        dest="dir",
        type=str,
        default=None,
        help="Change directory to the one specified before doing anything else.",
    )

    kernel_names = ",".join(GPyModel.kernel_names)
    acq_fn_names = ",".join(GPyModel.acquisition_fn_names)

    model_parser = argparse.ArgumentParser(add_help=None)
    model_parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help=f"Specifies the GP kernel. Allowed values are: {kernel_names}",
    )
    model_parser.add_argument(
        "--acquisition_fn",
        type=str,
        default="ei",
        help=f"Specifies the acquisition function. Allowed values are: {acq_fn_names}",
    )

    # TODO: specify this properly
    sp = parser.add_subparsers(dest="bopt")
    sp.required = True

    sp_init = sp.add_parser(
        "init", help="Initializes a new experiment, ready to run.",
        parents=[cd_parser]
    )

    # TODO: parallel evaluation

    sp_init.add_argument(
        "--result_parser",
        type=str,
        default="bopt.LastLineLastWordParser",
        help="Module path to the result parser.",
    )

    sp_init.add_argument("--runner", type=str, default="local", help="Runner type.")
    sp_init.add_argument("--param", action="append", help="Hyperparameter")
    sp_init.add_argument("command", type=str, help="Command to run.")
    sp_init.add_argument("arguments", type=str, nargs="*", help="Default arguments.")
    sp_init.set_defaults(func=init.run)

    sp_jobstat = sp.add_parser("job", help="Retrieve job status.",
            parents=[cd_parser])
    sp_jobstat.add_argument(
        "--recursive",
        type=bool,
        default=True,
        help="Search subdirectories for job files",
        required=False,
    )
    sp_jobstat.add_argument("JOB_ID", type=int, default=".",
            help="Job to search for.")
    sp_jobstat.set_defaults(func=jobstat.run)

    sp_expstat = sp.add_parser(
        "exp", help="Overview status of an experiment.", parents=[cd_parser]
    )
    sp_expstat.set_defaults(func=exp.run)

    sp_web = sp.add_parser("web", help="Starts the web interface.",
            parents=[cd_parser])
    sp_web.add_argument(
        "--port", type=int, default=5500, help="Port to run the web interface on."
    )
    sp_web.set_defaults(func=web.run)

    sp_run = sp.add_parser(
        "run", help="Runs an experiment.", parents=[cd_parser, model_parser]
    )
    sp_run.add_argument(
        "--n_iter", type=int, default=20, help="Number of iterations to run"
    )
    sp_run.add_argument(
        "--n_parallel",
        type=int,
        default=1,
        help="Number of instances to launch in parallel.",
    )
    sp_run.set_defaults(func=run.run)

    sp_run_single = sp.add_parser(
        "run-single",
        help="Pick the next point and run it, combines suggest & manual-run.",
        parents=[cd_parser, model_parser],
    )
    sp_run_single.add_argument(
        "--n_parallel",
        type=int,
        default=1,
        help="Number of instances to launch in parallel.",
    )
    sp_run_single.set_defaults(func=run_single.run)

    sp_plot = sp.add_parser(
        "plot", help="Generate plots for a given experiment.", parents=[cd_parser]
    )
    sp_plot.set_defaults(func=plot.run)

    sp_suggest = sp.add_parser(
        "suggest",
        help="Suggests one next point for evaluation.",
        parents=[cd_parser, model_parser],
    )
    sp_suggest.set_defaults(func=suggest.run)

    sp_manual_run = sp.add_parser(
        "manual-run",
        help="Run the next evaluation with manually given hyperparameters.",
        parents=[cd_parser],
    )
    sp_manual_run.set_defaults(func=manual_run.run)

    sp_debug = sp.add_parser(
        "debug", help="Load an experiment and start iPDB", parents=[cd_parser]
    )
    sp_debug.set_defaults(func=debug.run)

    sp_clean = sp.add_parser(
        "clean", help="Delete all samples, kill all jobs.", parents=[cd_parser]
    )
    sp_clean.set_defaults(func=clean.run)

    parsed, unknown = parser.parse_known_args()
    # this is an 'internal' method which returns 'parsed', the same as what
    # parse_args() would return and 'unknown', the remainder of that the
    # difference to parse_args() is that it does not exit when it finds
    # redundant arguments

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # TODO: fuj
            sp_manual_run.add_argument(arg.split("=")[0], type=str)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
