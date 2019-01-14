import bopt

hyperparameters = [
    bopt.Hyperparameter("gamma", bopt.Float(0, 1)),
    bopt.Hyperparameter("epsilon", bopt.Float(0, 1)),
    bopt.Hyperparameter("epsilon_final", bopt.Float(-1, 2)),
]

meta_dir = "results/rl-monte-carlo"
sge_runner = bopt.LocalRunner(
        meta_dir,
        "./.venv/bin/python",
        ["./experiments/rl/monte_carlo.py"],
        bopt.LastLineLastWordParser()
        )


experiment = bopt.Experiment(meta_dir, hyperparameters, sge_runner)
loop = bopt.OptimizationLoop(hyperparameters)

loop.run(experiment, n_iter=20)
