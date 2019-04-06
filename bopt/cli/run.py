import os
import sys
import yaml
import logging
import psutil
import time

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml, acquire_lock


def run(args) -> None:
    handle_cd(args)

    with ensure_meta_yml():
        logging.info("Found existing meta.yml, resuming experiment.")

        n_started = 0

        while n_started < args.n_iter:
            with acquire_lock():
                experiment = bopt.Experiment.deserialize()

                num_running = len([s for s in experiment.samples
                                   if s.job and not s.result and not s.job.is_finished()])

                if num_running < args.n_parallel:
                    experiment.collect_results()

                    fitted_model, sample = experiment.run_next()
                    logging.info("[{}/{}] Started a new evaluation"\
                           .format(n_started, args.n_iter))

                    n_started += 1

                    experiment.serialize()

                    if not sample.job and \
                            sample.collect_flag != bopt.CollectFlag.WAITING_FOR_SIMILAR:
                        logging.error("Created invalid sample without a job "
                            "(should have WAITING_FOR_SIMILAR).")

            psutil.wait_procs(psutil.Process().children(), timeout=0.01)
            time.sleep(args.sleep)
