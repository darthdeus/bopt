import unittest
import warnings

import numpy as np
import bopt
from deepdiff import DeepDiff


def test_exp1():
    from bopt.models.gpy_model import GPyModel
    import GPy

    hyperparameters = [
        bopt.Hyperparameter("x", bopt.Float(0.1, 0.7)),
        bopt.Hyperparameter("pi", bopt.Integer(24, 42))
    ]

    runner = bopt.LocalRunner("/bin/bash", ["--help", "me"])

    experiment = bopt.Experiment(hyperparameters, runner, r"(\d+)")

    X = np.random.uniform(-3., 3., (20, 1))
    Y = np.sin(X) + np.random.randn(20, 1) * 0.05

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        acq_fn = bopt.ExpectedImprovement()
        kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1 = GPy.models.GPRegression(X, Y, kernel, normalizer=len(X) > 1)
        gpy_model = GPyModel(m1, acq_fn)

    sp1 = bopt.HyperparamValues.sample_params(hyperparameters)
    sp2 = bopt.HyperparamValues.sample_params(hyperparameters)

    p1 = bopt.HyperparamValues.mapping_from_vector(sp1, hyperparameters)
    p2 = bopt.HyperparamValues.mapping_from_vector(sp2, hyperparameters)

    manual_params = bopt.ModelParameters.for_manual_run()

    samples = [
        # TODO: add tests for predicting with unfinished jobs
        bopt.Sample(bopt.LocalJob(314), manual_params, p1, 1.0, 1.2,
            bopt.CollectFlag.WAITING_FOR_JOB),
        bopt.Sample(bopt.SGEJob(314), manual_params, p2, 1.0, 1.0,
            bopt.CollectFlag.WAITING_FOR_JOB)
    ]

    experiment.samples = samples

    return experiment

class TestToDictFromDict(unittest.TestCase):
    def test_experiment(self):
        experiment = test_exp1()

        dd = experiment.to_dict()
        deserialized = bopt.Experiment.from_dict(dd)

        diff = DeepDiff(experiment.to_dict(), deserialized.to_dict())
        self.assertDictEqual({}, diff)


if __name__ == "__main__":
    unittest.main()
