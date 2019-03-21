import unittest
import warnings

import numpy as np
import bopt
import GPy
from deepdiff import DeepDiff


def test_exp1():
    hyperparameters = [
        bopt.Hyperparameter("x", bopt.Float(0.1, 0.7)),
        bopt.Hyperparameter("pi", bopt.Integer(24, 42))
    ]

    runner = bopt.LocalRunner("/bin/bash", ["--help", "me"])

    experiment = bopt.Experiment(hyperparameters, runner)

    X = np.random.uniform(-3., 3., (20, 1))
    Y = np.sin(X) + np.random.randn(20, 1) * 0.05

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        acq_fn = bopt.ExpectedImprovement()
        kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m1 = GPy.models.GPRegression(X, Y, kernel, normalizer=len(X) > 1)
        gpy_model = bopt.GPyModel(m1, acq_fn)

    sp1 = bopt.JobParams.sample_params(hyperparameters)
    sp2 = bopt.JobParams.sample_params(hyperparameters)

    p1 = bopt.JobParams.mapping_from_vector(sp1, hyperparameters)
    p2 = bopt.JobParams.mapping_from_vector(sp2, hyperparameters)

    manual_params = bopt.ModelParameters.for_manual_run()

    samples = [
        # TODO: add tests for predicting with unfinished jobs
        bopt.Sample(bopt.LocalJob(314, p1), manual_params, 1.0, 1.2),
        bopt.Sample(bopt.SGEJob(314, p2), manual_params, 1.0, 1.0)
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
