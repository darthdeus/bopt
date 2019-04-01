import logging
import sys

from bopt.cli.util import handle_cd, acquire_lock
from bopt.experiment import Experiment


def run(args) -> None:
    handle_cd(args)

    with acquire_lock():
        experiment = Experiment.deserialize()
        experiment.collect_results()

        __import__('ipdb').set_trace()

        print("Debug finished")
