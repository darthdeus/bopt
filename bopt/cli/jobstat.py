import glob
import os
import re
import sys
import traceback

from bopt.experiment import Experiment
from bopt.cli.util import handle_cd, acquire_lock


def run(args) -> None:
    # TODO: this is completely outdated and broken at this point
    raise NotImplementedError()


    handle_cd(args)

    with acquire_lock():
        job_files = glob.glob(os.path.join("**", "job-*.yml"), recursive=True)

        pattern = ".*job-(\\d+).yml"

        matches = [(fname, re.match(pattern, fname)) for fname in job_files]
        job_ids = [
            (fname, int(m.groups()[0]))
            for fname, m in matches
            if m is not None and len(m.groups()) == 1
        ]

        if len(job_ids) == 0:
            print(f"No jobs found. Check that {args.dir} contains job results.")
            sys.exit(1)

        matched_job_ids = [
            (fname, job_id) for fname, job_id in job_ids if job_id == args.JOB_ID
        ]

        if len(matched_job_ids) == 0:
            print(f"Job with id {args.JOB_ID} not found in '{args.dir}'.")
            sys.exit(1)
        elif len(matched_job_ids) > 1:
            print(f"Found more than one job with id {args.JOB_ID} in '{args.dir}'.")
            sys.exit(1)

        assert len(matched_job_ids) == 1

        fname, job_id = matched_job_ids[0]

        experiment = Experiment.deserialize()
        experiment.collect_results()

        # TODO: this is most likely not needed.
        job = experiment.runner.deserialize_job(job_id)

        is_finished = job.is_finished()

        print(f"Status:\t\t{job.status()}")
        if is_finished:
            try:
                if job.is_success():
                    print(f"Final result:\t{job.final_result()}")
                else:
                    print(f"Error:\t{job.err()}")
            except ValueError as e:
                traceback.print_exc()

        print()
        print(f"Parameters:")
        for key, value in job.run_parameters.items():
            print(f"\t{key}: {value}")

        print("RAW OUTPUT:")
        print("----------------------------------------------")
        print(job.get_job_output())
