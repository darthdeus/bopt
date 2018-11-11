import unittest
import numpy as np

from kernels import Linear


class TestKernel(unittest.TestCase):
    def test_linear(self):
        linear = Linear()
        self.assertAlmostEqual(6, linear([2], [3])[0].item())

        a = np.array([2])
        b = np.array([3])
        self.assertAlmostEqual(6, linear(a, b)[0].item())

        simple_expansion = linear(np.array([[2, 5]]), np.array([[3, 7]]))
        self.assertAlmostEqual(2*3 + 5*7, simple_expansion[0].item())

        multiple = linear(np.array([[2, 5], [3, 1]]), np.array([[3, 7], [2, 1]]))
        self.assertAlmostEqual(2*3 + 5*7, multiple[0, 0].item())
        self.assertAlmostEqual(2*2 + 5*1, multiple[0, 1].item())
        self.assertAlmostEqual(3*3 + 1*7, multiple[1, 0].item())
        self.assertAlmostEqual(3*2 + 1*1, multiple[1, 1].item())
