# TODO: get rid of psutil?
import psutil
import logging
import sys

from bopt.cli.util import handle_cd, acquire_lock
from bopt.experiment import Experiment


def run(args) -> None:
    handle_cd(args)

    with acquire_lock():
        experiment = Experiment.deserialize()
        experiment.collect_results()

        if args.r:
            for sample in experiment.samples:
                if sample.result:
                    print(sample.result)

            return


        print("Hyperparameters:")
        for param in experiment.hyperparameters:
            print(f"\t{param}")

        best_res = None
        best_sample = None

        ok_samples = []

        for sample in experiment.samples:
            if sample.result:
                try:
                    if best_res is None or (sample.result and sample.result > best_res):
                        best_res = sample.result
                        best_sample = sample
                except ValueError:
                    # TODO: cleanup checks
                    job_id = sample.job.job_id if sample.job else "NOJOB_ERR"
                    logging.error("Sample {} failed to parse".format(job_id))
                    continue

            ok_samples.append(sample)

        if best_sample:
            best_job_id = best_sample.job.job_id if best_sample.job else "NO_JOB"

            print("\nBEST (id={}): {}".format(best_job_id, best_res))
            assert best_sample is not None

            if best_sample.job:
                run_str = experiment.runner.script_path + " \\\n   "
                for h, v in best_sample.hyperparam_values.mapping.items():
                    if isinstance(v, float):
                        v = round(v, 2)
                    run_str += " --{}={}".format(h.name, v)

                print(run_str)
                print()

        print("Evaluations:")
        for sample in ok_samples:
            job = sample.job

            proc_stats = ""
            if job and psutil.pid_exists(job.job_id):
                process = psutil.Process(job.job_id)
                mem = process.memory_info()
                proc_stats += f"Process:{process.status()}"
                proc_stats += f", cpu={process.cpu_percent()}"
                # TODO fix this on osx, shared={mem.shared}"
                proc_stats += f", rss={mem.rss}, vms={mem.vms}"

            print(f"{sample}\t{proc_stats}")
