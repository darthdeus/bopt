import myopt as opt

params = [
    opt.Hyperparameter("x", opt.Float(0, 1)),
    opt.Hyperparameter("y", opt.Float(0, 1)),
]

meta_dir = "results/meta-dir"
sge_runner = opt.SGERunner(meta_dir, "./test.sh", ["default", "--argument=3"])

experiment = opt.Experiment("results/meta-dir", params, sge_runner)
