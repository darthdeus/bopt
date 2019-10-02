import datetime
import unittest
import warnings
import argparse

import numpy as np
import bopt

import tests.opt_functions as test_opt

class TestOptFunctions(unittest.TestCase):
    def test_opt_functions(self):
        funcs = test_opt.get_opt_test_functions()

        for f in funcs:
            print(f.name)
