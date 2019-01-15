# just to make sure the jobs don't fail on missing imports
import gym
import time

import psutil
import bopt

hyperparameters = [
    bopt.Hyperparameter("x", bopt.Float(1, 5)),
    bopt.Hyperparameter("y", bopt.Float(1, 5)),
]

meta_dir = "results/simple-function"
runner = bopt.LocalRunner(
        meta_dir,
        "./.venv/bin/python",
        ["./experiments/simple_function.py"],
        bopt.LastLineLastWordParser()
        )

n_iter = 20
done = 0

experiment = bopt.Experiment(meta_dir, hyperparameters, runner)
loop = bopt.OptimizationLoop(hyperparameters)

loop.run(experiment, n_iter=20)
