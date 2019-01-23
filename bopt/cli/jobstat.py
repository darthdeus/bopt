import glob
import os
import re
import sys
import traceback
from bopt.hyperparameters import Experiment


def run(args):
    job_files = glob.glob(
        os.path.join(args.META_DIR, "**", "job-*.yml"), recursive=True
    )

    pattern = ".*job-(\\d+).yml"

    matches = [(fname, re.match(pattern, fname)) for fname in job_files]
    job_ids = [
        (fname, int(m.groups()[0]))
        for fname, m in matches
        if m is not None and len(m.groups()) == 1
    ]

    if len(job_ids) == 0:
        print(f"No jobs found. Check that {args.META_DIR} contains job results.")
        sys.exit(1)

    matched_job_ids = [
        (fname, job_id) for fname, job_id in job_ids if job_id == args.JOB_ID
    ]

    if len(matched_job_ids) == 0:
        print(f"Job with id {args.JOB_ID} not found in '{args.META_DIR}'.")
        sys.exit(1)
    elif len(matched_job_ids) > 1:
        print(f"Found more than one job with id {args.JOB_ID} in '{args.META_DIR}'.")
        sys.exit(1)

    assert len(matched_job_ids) == 1

    fname, job_id = matched_job_ids[0]

    meta_dir = os.path.dirname(fname)
    experiment = Experiment.deserialize(meta_dir)

    # TODO: this is most likely not needed.
    job = experiment.runner.deserialize_job(meta_dir, job_id)

    is_finished = job.is_finished()

    print(f"Status:\t\t{job.status_str()}")
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