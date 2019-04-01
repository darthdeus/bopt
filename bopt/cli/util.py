import os
import sys
import filelock


def handle_cd(args):
    if args.dir is not None:
        print(f"Changing directory to {args.dir}", file=sys.stderr)
        os.chdir(args.dir)


def acquire_lock():
    return filelock.FileLock(".lockfile")


class handle_cd_revertible:
    def __init__(self, args) -> None:
        assert args.dir
        self.args = args

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.args.dir)

    def __exit__(self, type, value, traceback):
        os.chdir(self.old)


class ensure_meta_yml:
    def __enter__(self):
        if os.path.exists("meta.yml"):
            print("Found existing meta.yml.", file=sys.stderr)
        else:
            print(f"No meta.yml found in {os.path.abspath(os.curdir)}", file=sys.stderr)
            sys.exit(1)

    def __exit__(self, type, value, traceback):
        # print(type, value, traceback)
        pass

