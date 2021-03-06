# import sys
# from bopt.experiment import Experiment
# from bopt.cli.util import handle_cd_revertible, acquire_lock
#
# args = sys.argv[2:]
# experiments = []
#
# for exp_dir in args:
#     with handle_cd_revertible(exp_dir), acquire_lock():
#         print(exp_dir)
#         experiment = Experiment.deserialize()
#         experiments.append(experiment)
#
# print(len(experiments))
#
# import sys
# sys.exit(0)


import argparse
import os
import sys

from bopt.experiment import Experiment
from bopt.cli import (
    exp,
    init,
    jobstat,
    manual_run,
    plot,
    run,
    run_single,
    suggest,
    web,
    debug,
    clean,
    convert_meta
)


def main():
    run_main(sys.argv[1:])


def run_main(args):
    parser = argparse.ArgumentParser(prog="bopt")
    parser.set_defaults()

    cd_parser = argparse.ArgumentParser(add_help=None)
    cd_parser.add_argument(
        "-C",
        dest="dir",
        type=str,
        default=".",
        help="Change directory to the one specified before doing anything else.",
    )

    kernel_names = ",".join(Experiment.kernel_names)
    acq_fn_names = ",".join(Experiment.acquisition_fn_names)

    sp = parser.add_subparsers(dest="bopt")
    sp.required = True

    ### bopt init
    sp_init = sp.add_parser(
        "init", help="Initializes a new experiment, ready to run.",
        parents=[cd_parser]
    )

    sp_init.add_argument("--task", type=str, required=True, help="Task name.")
    sp_init.add_argument("--batch", type=str, required=False, help="Batch name.")
    sp_init.add_argument("--runner", type=str, default="local", help="Runner type.")
    sp_init.add_argument("--param", action="append", help="Hyperparameter", default=[])
    # TODO: default=[]
    sp_init.add_argument("--qsub", action="append", help="Arguments for qsub.")

    use_new_cli = os.getenv("USE_NEW_CLI", False)

    if use_new_cli:
        import bopt.gp_config as gp_config
        for config_param in gp_config.config_params:
            sp_init.add_argument("--{}".format(config_param.name),
                                 type=config_param.type,
                                 default=config_param.default,
                                 action=config_param.action,
                                 help=config_param.help)

    else:
        sp_init.add_argument("--kernel", type=str, default="Mat52",
                             help=f"Specifies the GP kernel. Allowed values are: {kernel_names}")

        sp_init.add_argument("--acquisition_fn", type=str, default="ei",
                             help=f"Specifies the acquisition function. Allowed values are: {acq_fn_names}")

        sp_init.add_argument("--ard", type=int, default=1,
                             help="Toggles automatic relevance determination (one lengthscale per hyperparameter).")

        sp_init.add_argument("--fit-mean", type=int, default=1,
                             help="When enabled the mean function is fit during kernel optimization. "
                             "Otherwise it is set to zero.")

        sp_init.add_argument("--gamma-prior", type=int, default=1,
                             help="When enabled, kernel parameters will use a Gamma prior "
                             "instead of a hard constraint.")
        sp_init.add_argument("--gamma-a", type=float, default=1.0,
                             help="The shape parameter of the Gamma prior.")
        sp_init.add_argument("--gamma-b", type=float, default=0.1,
                             help="The inverse rate parameter of the Gamma prior.")

        sp_init.add_argument("--informative-prior", type=int, default=1,
                             help="When enabled, kernel parameters use an informative Gamma prior on lengthscale.")
        sp_init.add_argument("--discrete-rounding", type=int, default=1,
                             help="Round discrete hyperparameters in the kernel.")

        sp_init.add_argument("--acq-xi", type=float, default=0.001,
                             help="The xi parameter of the acquisition functions")
        sp_init.add_argument("--acq-n-restarts", type=int, default=25,
                             help="Number of restarts when optimizing the acquisition function.")

        sp_init.add_argument("--num-optimize-restarts", type=int, default=10,
                             help="Number of restarts during kernel optimization.")
        sp_init.add_argument("--random-search-only", action="store_true", default=False,
                             help="Only use random search when picking new hyperparameters.", required=False)

    sp_init.add_argument("--manual-arg-fname", action="append", default=[],
                         help="Path to a file containing values for the manual argument.")

    sp_init.add_argument("command", type=str, help="Command to run.")
    sp_init.add_argument("arguments", type=str, nargs="*", help="Default arguments.")

    sp_init.set_defaults(func=init.run)


    ### bopt jobstat

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


    ### bopt convert-meta

    sp_convert = sp.add_parser("convert-meta",
                               help="Convert between different meta file formats",
                               parents=[cd_parser])
    sp_convert.add_argument("--to-json", action="store_true", default=False, help="Convert existing format to JSON.")
    sp_convert.add_argument("--to-yaml", action="store_true", default=False, help="Convert existing format to YAML.")
    sp_convert.add_argument("--keep", action="store_true", default=False, help="Keep the old file.")
    sp_convert.set_defaults(func=convert_meta.run)


    ### bopt exp

    sp_expstat = sp.add_parser(
        "exp", help="Overview status of an experiment.", parents=[cd_parser]
    )
    sp_expstat.add_argument("-r", action="store_true", default=False, help="Print only results.", required=False)
    sp_expstat.add_argument("-b", action="store_true", default=False, help="Print only best.", required=False)
    sp_expstat.set_defaults(func=exp.run)


    ### bopt web

    sp_web = sp.add_parser("web", help="Starts the web interface.",
                           parents=[cd_parser])
    sp_web.add_argument(
        "--port", type=int, default=5500, help="Port to run the web interface on."
    )
    sp_web.set_defaults(func=lambda args: web.run("single", args))


    ### bopt multiweb

    sp_multiweb = sp.add_parser("multiweb", help="Starts the web interface for multiple experiments",
                                parents=[cd_parser])
    # TODO: dupe
    sp_multiweb.add_argument(
        "--port", type=int, default=5500, help="Port to run the web interface on."
    )
    sp_multiweb.add_argument("experiments", nargs="+")
    sp_multiweb.set_defaults(func=lambda args: web.run("multi", args))


    ### bopt run

    sp_run = sp.add_parser("run", help="Runs an experiment.", parents=[cd_parser])
    sp_run.add_argument("-c", action="store_true", default=False, help="Continue up to maximum n_iter.", required=False)
    sp_run.add_argument(
        "--n_iter", type=int, default=20, help="Number of iterations to run."
    )
    sp_run.add_argument(
        "--sleep", type=float, default=1.0, help="Number of seconds to sleep between iterations."
    )
    sp_run.add_argument(
        "--n_parallel",
        type=int,
        default=1,
        help="Number of instances to launch in parallel.",
    )
    sp_run.set_defaults(func=run.run)


    ### bopt run-single

    sp_run_single = sp.add_parser(
        "run-single",
        help="Pick the next point and run it, combines suggest & manual-run.",
        parents=[cd_parser],
    )
    sp_run_single.add_argument(
        "--n_parallel",
        type=int,
        default=1,
        help="Number of instances to launch in parallel.",
    )
    sp_run_single.set_defaults(func=run_single.run)


    ### bopt plot

    sp_plot = sp.add_parser(
        "plot", help="Generate plots for a given experiment.", parents=[cd_parser]
    )
    sp_plot.set_defaults(func=plot.run)


    ### bopt suggest

    sp_suggest = sp.add_parser(
        "suggest",
        help="Suggests one next point for evaluation.",
        parents=[cd_parser],
    )
    sp_suggest.set_defaults(func=suggest.run)


    ### bopt manual-run

    sp_manual_run = sp.add_parser(
        "manual-run",
        help="Run the next evaluation with manually given hyperparameters.",
        parents=[cd_parser],
    )
    sp_manual_run.set_defaults(func=manual_run.run)


    ### bopt debug

    sp_debug = sp.add_parser(
        "debug", help="Load an experiment and start iPDB", parents=[cd_parser]
    )
    sp_debug.set_defaults(func=debug.run)


    ### bopt clean

    sp_clean = sp.add_parser(
        "clean", help="Delete all samples, kill all jobs.", parents=[cd_parser]
    )
    sp_clean.set_defaults(func=clean.run)

    _, unknown = parser.parse_known_args(args)
    # this is an 'internal' method which returns 'parsed', the same as what
    # parse_args() would return and 'unknown', the remainder of that the
    # difference to parse_args() is that it does not exit when it finds
    # redundant arguments

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # TODO: fuj
            sp_manual_run.add_argument(arg.split("=")[0], type=str)

    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main()
