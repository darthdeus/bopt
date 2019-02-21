import numpy as np
import unittest
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

    # m1 = GPy.models.GPRegression()
    # m2 = GPy.models.sparse_GP_regression_1D(optimize=True, plot=False)

    X = np.random.uniform(-3., 3., (20, 1))
    Y = np.sin(X) + np.random.randn(20, 1) * 0.05

    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m1 = GPy.models.GPRegression(X, Y, kernel)
    gpy_model = bopt.GPyModel(m1)

    samples = [
        bopt.Sample({ "foo": "bar" },
            bopt.LocalJob(314, { "job": "params" }), gpy_model),

        bopt.Sample({ "foo": "goo" },
            bopt.SGEJob(314, { "job": "params" }), None)
    ]

    experiment.samples = samples

    return experiment

class TestToDictFromDict(unittest.TestCase):
    def test_experiment(self):
        experiment = test_exp1()

        deserialized = bopt.Experiment.from_dict(experiment.to_dict())

        self.assertDictEqual({}, DeepDiff(experiment, deserialized))


if __name__ == "__main__":
    unittest.main()
