import os


def handle_cd(args):
    if args.dir is not None:
        os.chdir(args.dir)

