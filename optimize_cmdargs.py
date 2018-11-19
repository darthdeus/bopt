import sys
from myopt.bayesian_optimization import Float, Integer, bo_minimize

arg_bounds = [
    ("alpha", Float(0.05, 1)),
    ("epsilon", Float(0.03, 0.8)),
]


def f(x):
    from subprocess import Popen, PIPE

    args = [f"--{name}={x[i]}" for i, (name, _) in enumerate(arg_bounds)]
    args.append("--episodes=1000")

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


if __name__ == '__main__':
    result = bo_minimize(f, [bound for _, bound in arg_bounds], n_iter=30)

    print("\n\n\n\n****************************")
    print(result)
