import logging

import bopt
from bopt.cli.util import handle_cd_revertible, ensure_meta_yml, acquire_lock


def run(args) -> None:
    with handle_cd_revertible(args):
        with acquire_lock(), ensure_meta_yml():
            experiment = bopt.Experiment.deserialize()
            experiment.collect_results()

            assert args.n_parallel > 0

            for i in range(args.n_parallel):
                logging.info("Starting {}/{}".format(i, args.n_parallel))
                experiment.run_next()
