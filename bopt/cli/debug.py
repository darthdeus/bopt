import logging
import sys

from bopt.cli.util import handle_cd_revertible, acquire_lock
from bopt.experiment import Experiment


def run(args) -> None:
    with handle_cd_revertible(args):
        with acquire_lock():
            experiment = Experiment.deserialize()
            experiment.collect_results()

            import ipdb
            ipdb.set_trace()

            print("Debug finished")
