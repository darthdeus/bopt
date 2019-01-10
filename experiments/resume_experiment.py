from myopt import Experiment, SGEJob

job = Experiment.deserialize("results/meta-dir").runner.start({})

