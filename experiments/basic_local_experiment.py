import bopt

params = [
    bopt.Hyperparameter("x", bopt.Float(0, 1)),
    bopt.Hyperparameter("y", bopt.Float(0, 1)),
]

meta_dir = "results/local"
sge_runner = bopt.LocalRunner(meta_dir, "./test.sh", ["default", "--argument=3"])

experiment = bopt.Experiment(meta_dir, params, sge_runner)

experiment.runner.start({"a": 3})
