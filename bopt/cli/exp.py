# TODO: get rid of psutil?
import psutil

from bopt.cli.util import handle_cd
from bopt.experiment import Experiment


def run(args) -> None:
    handle_cd(args)

    experiment = Experiment.deserialize(".")

    print("Hyperparameters:")
    for param in experiment.hyperparameters:
        print(f"\t{param}")

    print("BEST:")

    best_res = None
    best_sample = None

    for sample in experiment.samples:
        if sample.job.is_finished():
            res = sample.get_result(".")

            if best_res is None or res > best_res:
                best_res = res
                best_sample = sample

    print("objective: {}".format(best_res))
    print(best_sample.job.run_parameters)
    print()

    print("Evaluations:")
    for sample in experiment.samples:
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
