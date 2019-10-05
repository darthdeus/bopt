import os
import shutil
import pathlib
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

        from bopt.cli.cli import run_main

        base_path = "tmp/test_exps"

        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)

        for f in funcs:
            print(f.name)
            run_main(["init", "-C", "{}/{}".format(base_path, f.name), "ls"])
