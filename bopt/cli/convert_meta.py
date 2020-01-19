import sys

from bopt.gp_config import GPConfig
from bopt.cli.util import handle_cd_revertible, acquire_lock

def run(args):
    with handle_cd_revertible(args.dir):
        with acquire_lock():
            if args.to_json:
                with open("meta.yml", "r") as f_src:
                    with open("meta.json", "w") as f_dst:
                        import yaml
                        import json

                        data = yaml.load(f_src, Loader=yaml.Loader)
                        if isinstance(data["gp_config"], GPConfig):
                            data["gp_config"] = data["gp_config"].to_dict()

                        json.dump(data, f_dst)

            elif args.to_yaml:
                raise NotImplementedError()
            else:
                print("Must provide either --to-yaml or --to-json", file=sys.stderr)
                sys.exit(1)
