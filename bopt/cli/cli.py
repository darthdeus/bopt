import sys
import os
import psutil
import inspect
import argparse

from bopt.cli import jobstat, expstat, web, new_exp, run_exp, init_exp


def main():
    parser = argparse.ArgumentParser(prog="bopt")
    parser.set_defaults()

    # TODO: specify this properly
    sp = parser.add_subparsers(dest="bopt")
    sp.required = True

    sp_new = sp.add_parser("new", help="Create a new experiment.")
    sp_init = sp.add_parser("init", help="Initializes a new experiment, ready to run.")
    sp_expstat = sp.add_parser("exp", help="Overview status of an experiment.")
    sp_jobstat = sp.add_parser("job", help="Retrieve job status.")
    sp_web = sp.add_parser("web", help="Starts the web interface.")
    sp_run = sp.add_parser("run", help="Runs an experiment.")

    sp_init.add_argument(
        "--result_parser",
        type=str,
        default="bopt.LastLineLastWordParser",
        help="Module path to the result parser.",
    )
    sp_init.add_argument("--runner", type=str, default="local", help="Runner type.")
    sp_init.add_argument("--param", action="append", help="Hyperparameter")
    sp_init.add_argument(
        "--dir",
        type=str,
        help="Path to the directory where the experiment data is stored.",
    )
    sp_init.add_argument("command", type=str, help="Command to run.")
    sp_init.add_argument("arguments", type=str, nargs="*", help="Default arguments.")
    sp_init.set_defaults(func=init_exp.run)

    sp_new.add_argument(
        "DIR",
        type=str,
        help="Path to the directory where the experiment should be stored, both its configuration and results.",
    )
    sp_new.set_defaults(func=new_exp.run)

    sp_run.add_argument(
        "DIR",
        type=str,
        help="Path to the directory where the experiment data is stored. This is the same path used when creating using \`bopt new\`.",
    )
    sp_run.set_defaults(func=run_exp.run)

    sp_jobstat.add_argument(
        "--recursive",
        type=bool,
        default=True,
        help="Search subdirectories for job files",
        required=False,
    )
    sp_jobstat.add_argument(
        "META_DIR", type=str, default=".", help="Directory with the results.", nargs="?"
    )
    sp_jobstat.add_argument("JOB_ID", type=int, default=".", help="Job to search for.")
    sp_jobstat.set_defaults(func=jobstat.run)

    sp_expstat.add_argument("meta_dir", type=str, help="Directory with the results.")
    sp_expstat.set_defaults(func=expstat.run)

    sp_web.add_argument("meta_dir", type=str, help="Directory with the results.")
    sp_web.add_argument("--web", action="store_true")
    sp_web.add_argument(
        "--port", type=int, default=5500, help="Port to run the web interface on."
    )
    sp_web.set_defaults(func=web.run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
