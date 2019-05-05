import os
import glob
import logging
import sys

from bopt.cli.util import handle_cd, acquire_lock
from bopt.experiment import Experiment


def run(args) -> None:
    handle_cd(args)

    with acquire_lock():
        experiment = Experiment.deserialize()

        for sample in experiment.samples:
            if sample.job and not sample.job.is_finished():
                sample.job.kill()

        experiment.samples = []
        experiment.serialize()

        for f in (glob.glob("output/*") + glob.glob("plots/*")):
            os.remove(f)
