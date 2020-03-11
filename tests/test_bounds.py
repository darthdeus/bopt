import numpy as np
import unittest

from bopt import Integer, Float, LogscaleInt, LogscaleFloat, Discrete


class TestOptFunctions(unittest.TestCase):
    def test_int_bins(self):
        self.assertListEqual(
            [0., 1., 2., 3., 4., 5.],
            Integer(0, 5, -1).compute_bins()
        )

        self.assertListEqual(
            [0., 2., 4.],
            Integer(0, 4, 3).compute_bins()
        )

        self.assertListEqual(
            [0., 1., 2.],
            Integer(0, 2, 3).compute_bins()
        )

    def test_float_bins(self):
        self.assertListEqual(
            [0., 2., 4.],
            Float(0, 4, 3).compute_bins()
        )

        self.assertListEqual(
            [0., 1., 2.],
            Float(0, 2, 3).compute_bins()
        )

        tenbin = Float(0, 2, 10).compute_bins()

        self.assertEqual(0., tenbin[0])
        self.assertAlmostEqual(0.222222222, tenbin[1], places=5)
        self.assertEqual(2., tenbin[9])

    def test_discrete_bins(self):
        self.assertListEqual(
            [0., 1., 2.],
            Discrete(["sigmoid", "tanh", "relu"]).compute_bins()
        )

    def test_discrete_maybe_round(self):
        d = Discrete(["sigmoid", "tanh", "relu"])

        for x in np.arange(-10.0, 0.49, 0.1):
            self.assertEqual(0., d.maybe_round(np.array([x])), x)
        for x in np.arange(0.51, 1.49, 0.1):
            self.assertEqual(1., d.maybe_round(np.array([x])), x)
        for x in np.arange(1.51, 10, 0.1):
            self.assertEqual(2., d.maybe_round(np.array([x])), x)

    def test_int_maybe_round(self):
        i = Integer(0, 5, -1)
        for x in np.arange(-10.0, 0.49, 0.1):
            self.assertEqual(0., i.maybe_round(np.array([x])), x)
        for x in np.arange(0.51, 1.49, 0.1):
            self.assertEqual(1., i.maybe_round(np.array([x])), x)
        for x in np.arange(4.51, 10., 0.1):
            self.assertEqual(5., i.maybe_round(np.array([x])), x)

        i = Integer(0, 4, 3)
        for x in np.arange(-10.0, 0.99, 0.1):
            self.assertEqual(0., i.maybe_round(np.array([x])), x)
        for x in np.arange(1.01, 2.99, 0.1):
            self.assertEqual(2., i.maybe_round(np.array([x])), x)
        for x in np.arange(3.01, 10., 0.1):
            self.assertEqual(4., i.maybe_round(np.array([x])), x)

    def test_float_maybe_round(self):
        f = Float(0., 4., 3)
        for x in np.arange(-10.0, 0.99, 0.1):
            self.assertEqual(0., f.maybe_round(np.array([x])), x)
        for x in np.arange(1.01, 2.99, 0.1):
            self.assertEqual(2., f.maybe_round(np.array([x])), x)
        for x in np.arange(3.01, 10., 0.1):
            self.assertEqual(4., f.maybe_round(np.array([x])), x)

        f = Float(0., 4., 10)
        for x in np.arange(-10.0, 0.22, 0.1):
            self.assertEqual(0., f.maybe_round(np.array([x])), x)
        for x in np.arange(0.24, 0.64, 0.1):
            self.assertAlmostEqual(0.4444, f.maybe_round(np.array([x])).item(), places=3, msg=x)
        for x in np.arange(3.88, 10., 0.1):
            self.assertEqual(4., f.maybe_round(np.array([x])), x)
