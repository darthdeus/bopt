import unittest
import numpy as np
import pytest

# from bopt.kernels import Linear


class TestKernel(unittest.TestCase):
    def test_meshgrid_replacement(self):
        a = np.array([1,2])
        b = np.array([4,5])

        c = a.reshape(-1, 1) + b.reshape(1, -1)

        aa, bb = np.meshgrid(a, b)
        d = aa + bb

        self.assertTrue(np.allclose(c, d))
