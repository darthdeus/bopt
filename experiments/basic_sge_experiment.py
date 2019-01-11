import bopt

params = [
    bopt.Hyperparameter("x", bopt.Float(0, 1)),
    bopt.Hyperparameter("y", bopt.Float(0, 1)),
]

meta_dir = "results/meta-dir"
sge_runner = bopt.SGERunner(meta_dir, "./test.sh", ["default", "--argument=3"])

experiment = bopt.Experiment("results/meta-dir", params, sge_runner)
