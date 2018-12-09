import myopt as opt


params = [
    opt.Hyperparameter("x", opt.Float(0, 1)),
    opt.Hyperparameter("y", opt.Float(0, 1)),
]

sge_runner = opt.SGERunner("./test.sh", ["hello", "world"])

experiment = opt.Experiment(params, sge_runner)
