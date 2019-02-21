import unittest
import bopt
from deepdiff import DeepDiff


def test_exp1():
    hyperparameters = [
            bopt.Hyperparameter("x", bopt.Float(0.1, 0.7)),
            bopt.Hyperparameter("pi", bopt.Integer(24, 42))
    ]

    runner = bopt.LocalRunner("/bin/bash", ["--help", "me"])

    experiment = bopt.Experiment(hyperparameters, runner)

    return experiment

class TestToDictFromDict(unittest.TestCase):
    def test_experiment(self):
        experiment = test_exp1()

        deserialized = bopt.Experiment.from_dict(experiment.to_dict())

        self.assertDictEqual({}, DeepDiff(experiment, deserialized))


if __name__ == "__main__":
    unittest.main()
