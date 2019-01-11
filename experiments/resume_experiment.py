from bopt import Experiment, SGEJob

job = Experiment.deserialize("results/meta-dir").runner.start({})

