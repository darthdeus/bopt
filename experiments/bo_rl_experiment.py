# just to make sure the jobs don't fail on missing imports
import gym
import time

import psutil
import bopt

hyperparameters = [
    bopt.Hyperparameter("gamma", bopt.Float(0, 1)),
    bopt.Hyperparameter("epsilon", bopt.Float(0, 1)),
]

meta_dir = "results/rl-monte-carlo"
sge_runner = bopt.LocalRunner(
        meta_dir,
        "./.venv/bin/python",
        ["./experiments/monte_carlo.py"],
        bopt.LastLineLastWordParser()
        )


n_iter = 20
done = 0

experiment = bopt.Experiment(meta_dir, hyperparameters, sge_runner)
loop = bopt.OptimizationLoop(bopt.SquaredExp(), hyperparameters, n_iter, None, True, 0,
                            bopt.expected_improvement)

experiment.runner.start({
    "gamma": 1,
    "epsilon": 1,
    # "sleep_when_done": 0
})

while done < n_iter:
    gamma, epsilon = loop.next()

    job = experiment.runner.start({
        "gamma": gamma,
        "epsilon": epsilon,
        # "sleep_when_done": 1
        })

    while not job.is_finished():
        psutil.wait_procs(psutil.Process().children(), timeout=0.01)
        time.sleep(1)

    # while True:
        # import multiprocessing
        #
        # print(multiprocessing.active_children())
        # time.sleep(1)




# for gamma in [.1, .3, .5, .7, .9, .95, .99]:
#     for epsilon in [.1, .3, .5, .7, .9, .95, .99]:
#         experiment.runner.start({
#             "gamma": gamma,
#             "epsilon": epsilon
#             })


# import time
# import numpy as np
# for gamma in np.arange(0.01, 0.99, step=0.25):
#     for epsilon in np.arange(0.01, 0.99, step=0.25):
#         experiment.runner.start({
#             "gamma": gamma,
#             "epsilon": epsilon
#             })
#         time.sleep(1)

# for gamma in [.1, .2, .4, .6, .8, .99]:
#     for epsilon in [.1, .4, .7, .9]:
#         experiment.runner.start({
#             "gamma": gamma,
#             "epsilon": epsilon
#             })
#         time.sleep(1)
