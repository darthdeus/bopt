import sys
from bopt.bayesian_optimization import Float, Integer, bo_maximize_parallel

arg_bounds = [
    ("alpha", Float(0.05, 1)),
    ("epsilon", Float(0.03, 0.8)),
]

import concurrent.futures

executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)


def f(x):
    from subprocess import Popen, PIPE

    args = [f"--{name}={x[i]}" for i, (name, _) in enumerate(arg_bounds)]
    args.append("--episodes=10000")

    cmd = ["python", "C:\\dev\\npfl122\\labs\\03\\q_learning.py", *args]
    print(f"Running: {cmd}")

    process = Popen(cmd, stdout=PIPE, cwd="C:\\dev\\npfl122\\labs\\03")
    (output, err) = process.communicate()

    exit_code = process.wait()
    print(f"Finished with: {exit_code}\n\n***")

    lines = output.splitlines()

    if len(lines) == 0:
        print("No output, exiting")
        sys.exit(1)

    # TODO: fuj, but works
    reward_str = lines[-1].decode("ascii").split(" ")[-1]
    reward = float(reward_str)

    print(output)
    print(f"*** Reward: {reward}")

    return reward


def parallel_f(x):
    return executor.submit(f, x)


if __name__ == '__main__':
    result = bo_maximize_parallel(parallel_f, [bound for _, bound in arg_bounds], n_iter=30, n_parallel=8)

    print("\n\n\n\n****************************")
    print(result)
