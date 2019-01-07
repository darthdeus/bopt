# just to make sure the jobs don't fail on missing imports
import gym
import myopt as opt

params = [
    opt.Hyperparameter("gamma", opt.Float(0, 1)),
    opt.Hyperparameter("epsilon", opt.Float(0, 1)),
]

meta_dir = "results/rl-monte-carlo"
sge_runner = opt.LocalRunner(
        meta_dir,
        "./.venv/bin/python",
        ["./experiments/monte_carlo.py"],
        opt.LastLineLastWordParser()
        )

experiment = opt.Experiment(meta_dir, params, sge_runner)


# for gamma in [.1, .3, .5, .7, .9, .95, .99]:
#     for epsilon in [.1, .3, .5, .7, .9, .95, .99]:
#         experiment.runner.start({
#             "gamma": gamma,
#             "epsilon": epsilon
#             })


import time
import numpy as np
for gamma in np.arange(0.01, 0.99, step=0.25):
    for epsilon in np.arange(0.01, 0.99, step=0.25):
        experiment.runner.start({
            "gamma": gamma,
            "epsilon": epsilon
            })
        time.sleep(1)

# for gamma in [.1, .2, .4, .6, .8, .99]:
#     for epsilon in [.1, .4, .7, .9]:
#         experiment.runner.start({
#             "gamma": gamma,
#             "epsilon": epsilon
#             })
#         time.sleep(1)
