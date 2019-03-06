import os
import sys


def handle_cd(args):
    if args.dir is not None:
        print(f"Changing directory to {args.dir}", file=sys.stderr)
        os.chdir(args.dir)

class ensure_meta_yml:
    def __enter__(self):
        if os.path.exists("meta.yml"):
            print("Found existing meta.yml.", file=sys.stderr)
        else:
            print(f"No meta.yml found in {os.curdir()}", file=sys.stderr)
            sys.exit(1)

    def __exit__(self, type, value, traceback):
        # print(type, value, traceback)
        pass

