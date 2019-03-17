# TODO: get rid of psutil?
import psutil
import logging

from bopt.cli.util import handle_cd
from bopt.experiment import Experiment


def run(args) -> None:
    handle_cd(args)

    experiment = Experiment.deserialize(".")

    print("Hyperparameters:")
    for param in experiment.hyperparameters:
        print(f"\t{param}")

    best_res = None
    best_sample = None

    ok_samples = []

    for sample in experiment.samples:
        if sample.job.is_finished():
            try:
                res = sample.get_result(".")

                if best_res is None or res > best_res:
                    best_res = res
                    best_sample = sample
            except ValueError:
                job_id = sample.job.job_id if sample.job else "NOJOB_ERR"
                logging.error("Sample {} failed to parse".format(job_id))
                continue

        ok_samples.append(sample)

    print("\nBEST (id={}): {}".format(best_sample.job.job_id, best_res))

    assert best_sample is not None

    if best_sample.job is not None:
        print(best_sample.job.run_parameters)
        print()

        print("Evaluations:")
        for sample in ok_samples:
            job = sample.job

            proc_stats = ""
            if psutil.pid_exists(job.job_id):
                process = psutil.Process(job.job_id)
                mem = process.memory_info()
                proc_stats += f"Process:{process.status()}"
                proc_stats += f", cpu={process.cpu_percent()}"
                # TODO fix this on osx, shared={mem.shared}"
                proc_stats += f", rss={mem.rss}, vms={mem.vms}"

            print(f"{sample}\t{proc_stats}")
