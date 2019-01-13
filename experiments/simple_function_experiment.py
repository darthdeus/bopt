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
sge_runner = bopt.LocalRunner(
        meta_dir,
        "./.venv/bin/python",
        ["./experiments/simple_function.py"],
        bopt.LastLineLastWordParser()
        )

n_iter = 20
done = 0

experiment = bopt.Experiment(meta_dir, hyperparameters, sge_runner)
loop = bopt.OptimizationLoop(hyperparameters, n_iter)


while done < n_iter:
    x, y = loop.next()

    job = experiment.runner.start({
        "x": x,
        "y": y,
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






# for x in [.1, .3, .5, .7, .9, .95, .99]:
#     for y in [.1, .3, .5, .7, .9, .95, .99]:
#         experiment.runner.start({
#             "x": x,
#             "y": y
#             })


# import time
# import numpy as np
# for x in np.arange(0.01, 0.99, step=0.25):
#     for y in np.arange(0.01, 0.99, step=0.25):
#         experiment.runner.start({
#             "x": x,
#             "y": y
#             })
#         time.sleep(1)

# for x in [.1, .2, .4, .6, .8, .99]:
#     for y in [.1, .4, .7, .9]:
#         experiment.runner.start({
#             "x": x,
#             "y": y
#             })
#         time.sleep(1)
