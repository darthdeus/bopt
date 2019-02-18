# TODO: get rid of psutil?
import psutil
from bopt.experiment import Experiment


def run(args) -> None:
    experiment = Experiment.deserialize(args.meta_dir)

    print("Hyperparameters:")
    for param in experiment.hyperparameters:
        print(f"\t{param}")

    print("Evaluations:")
    for job in experiment.evaluations:

        proc_stats = ""
        if psutil.pid_exists(job.job_id):
            process = psutil.Process(job.job_id)
            mem = process.memory_info()
            proc_stats += f"Process:{process.status()}"
            proc_stats += f", cpu={process.cpu_percent()}"
            # TODO fix this on osx, shared={mem.shared}"
            proc_stats += f", rss={mem.rss}, vms={mem.vms}"

        print(f"{job}\t{proc_stats}")
