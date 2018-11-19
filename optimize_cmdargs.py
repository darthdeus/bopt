import sys
from myopt.bayesian_optimization import Float, Integer, bo_minimize

arg_bounds = [
    ("alpha", Float(0.05, 2)),
    ("epsilon", Float(0.03, 0.8)),
    ("gamma", Float(0.8, 0.99))
]


def f(x):
    from subprocess import Popen, PIPE

    args = [f"--{name}={x[i]}" for i, (name, _) in enumerate(arg_bounds)]

    cmd = ["python", "C:\\dev\\npfl122\\labs\\03\\lunar_lander.py", *args]
    print(f"Running: {cmd}")

    process = Popen(cmd, stdout=PIPE, cwd="C:\\dev\\npfl122\\labs\\03")
    (output, err) = process.communicate()

    exit_code = process.wait()
    print(f"Finished with: {exit_code}\n\n***")

    lines = output.splitlines()

    if len(lines) == 0:
        print("No output, exiting")
        sys.exit(1)

    last_line = lines[-1]
    reward = float(last_line)

    print(output)
    print(f"*** Reward: {reward}")

    return reward


if __name__ == '__main__':
    result = bo_minimize(f, [bound for _, bound in arg_bounds], n_iter=10)

    print("\n\n\n\n****************************")
    print(result)
