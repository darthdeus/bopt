import os
import sys


def handle_cd(args):
    if args.dir is not None:
        print(f"Changing directory to {args.dir}", file=sys.stderr)
        os.chdir(args.dir)


# TODO: neco jako
# with bopt.ensure_meta_dir():
    # ...

