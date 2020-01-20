import sys

from bopt.gp_config import GPConfig
from bopt.cli.util import handle_cd_revertible, acquire_lock
from bopt.sample import maybe_datetime_to_timestamp


def flag_to_int(flag_or_int):
    if isinstance(flag_or_int, int):
        return flag_or_int
    else:
        return flag_or_int.value


def convert_collect_flag(sample):
    sample = dict(**sample)
    sample.update({
        "collect_flag": flag_to_int(sample["collect_flag"]),
        "created_at": maybe_datetime_to_timestamp(sample["created_at"]),
        "finished_at": maybe_datetime_to_timestamp(sample["finished_at"]),
        "collected_at": maybe_datetime_to_timestamp(sample["collected_at"]),
    })
    return sample


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

                        data["samples"] = [
                            convert_collect_flag(sample)
                            for sample in data["samples"]
                        ]

                        json.dump(data, f_dst)

            elif args.to_yaml:
                raise NotImplementedError()
            else:
                print("Must provide either --to-yaml or --to-json", file=sys.stderr)
                sys.exit(1)
