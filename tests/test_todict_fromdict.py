import unittest
import bopt

class TestToDictFromDict(unittest.TestCase):
    def test_experiment(self):
        hyperparameters = [
                bopt.Hyperparameter("x", bopt.Float(0.1, 0.7)),
                bopt.Hyperparameter("pi", bopt.Integer(24, 42))
        ]

        runner = bopt.LocalRunner("/bin/bash", ["--help", "me"])

        experiment = bopt.Experiment(hyperparameters, runner)

        deserialized = bopt.Experiment.from_dict(experiment.to_dict())

if __name__ == "__main__":
    unittest.main()
