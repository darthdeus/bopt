import unittest
import numpy as np
import pytest

from myopt.kernels import Linear


class TestKernel(unittest.TestCase):
    def test_meshgrid_replacement(self):
        a = np.array([1,2])
        b = np.array([4,5])

        c = a.reshape(-1, 1) + b.reshape(1, -1)

        aa, bb = np.meshgrid(a, b)
        d = aa + bb

        self.assertTrue(np.allclose(c, d))

    # def test_linear(self):
        # linear = Linear()
        # y = linear(np.array([[2]]), np.array([[3]]))[0].item()
        # print(y)
        # self.assertAlmostEqual(6, y)
        #
        # a = np.array([2])
        # b = np.array([3])
        # self.assertAlmostEqual(6, linear(a, b)[0].item())
        #
        # simple_expansion = linear(np.array([[2, 5]]), np.array([[3, 7]]))
        # self.assertAlmostEqual(2*3 + 5*7, simple_expansion[0].item())
        #
        # multiple = linear(np.array([[2, 5], [3, 1]]), np.array([[3, 7], [2, 1]]))
        # self.assertAlmostEqual(2*3 + 5*7, multiple[0, 0].item())
        # self.assertAlmostEqual(2*2 + 5*1, multiple[0, 1].item())
        # self.assertAlmostEqual(3*3 + 1*7, multiple[1, 0].item())
        # self.assertAlmostEqual(3*2 + 1*1, multiple[1, 1].item())
