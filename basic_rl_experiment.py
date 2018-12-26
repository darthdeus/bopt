import myopt as opt

params = [
    opt.Hyperparameter("gamma", opt.Float(0, 1)),
    opt.Hyperparameter("epsilon", opt.Float(0, 1)),
]

meta_dir = "results/rl-monte-carlo"
sge_runner = opt.LocalRunner(
        meta_dir,
        "/home/darth/projects/npfl122/.venv/bin/python",
        ["/home/darth/projects/npfl122/labs/02/monte_carlo.py"],
        opt.LastLineLastWordParser()
        )

experiment = opt.Experiment(meta_dir, params, sge_runner)


# for gamma in [.1, .3, .5, .7, .9, .95, .99]:
#     for epsilon in [.1, .3, .5, .7, .9, .95, .99]:
#         experiment.runner.start({
#             "gamma": gamma,
#             "epsilon": epsilon
#             })

for gamma in [.1, .99]:
    for epsilon in [.1, .5, .9]:
        experiment.runner.start({
            "gamma": gamma,
            "epsilon": epsilon
            })
