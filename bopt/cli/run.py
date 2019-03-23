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

        model_config = bopt.ModelConfig(args)

        while n_started < args.n_iter:
            with acquire_lock():
                experiment = bopt.Experiment.deserialize()

                if experiment.num_running() < args.n_parallel:
                    experiment.collect_results()

                    sample = experiment.run_single(model_config)

                    n_started += 1

                    experiment.serialize()

                    if sample.job:
                        logging.info("Started a new job {} with config {}" \
                                .format(sample.job.job_id, model_config))
                    else:
                        logging.error("Run loop created a sample without job.")

            psutil.wait_procs(psutil.Process().children(), timeout=0.01)
            time.sleep(args.sleep)

        # TODO: delete old code

        # for i in range(n_iter):
        #     job = self.run_single(model_config, meta_dir)
        #     logging.info("Started a new job {} with config {}".format(job.job_id, model_config))
        #
        #     start_time = time.time()
        #
        #     # psutil.wait_procs(psutil.Process().children(), timeout=0.01)
        #     while not job.is_finished():
        #         psutil.wait_procs(psutil.Process().children(), timeout=0.01)
        #         time.sleep(0.2)
        #
        #     end_time = time.time()
        #
        #     logging.info("Job {} finished after {}".format(job.job_id, end_time - start_time))
        #
        #     self.collect_results()
        #     self.serialize(meta_dir)
