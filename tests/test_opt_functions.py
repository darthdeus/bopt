import os
import shutil
import pathlib
import datetime
import unittest
import warnings
import argparse
import logging

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
            bx, by = f.bounds

            # TODO: this breaks the arg parser
            # run_main(["init", "-C", "{}/{}".format(base_path, f.name),
            #           "--param 'x:float:{}:{}'".format(bx.low, bx.high),
            #           "--param 'y:float:{}:{}'".format(by.low, by.high),
            #           "ls"])

            chdir = ["-C", "{}/{}".format(base_path, f.name)]

            init_args = ["init", *chdir,
                         "--task", "testing",
                         "--param", "x:float:{}:{}".format(bx.low, bx.high),
                         "--param", "y:float:{}:{}".format(by.low, by.high),
                         # TODO: this does not work "--random-search-only",
                         os.path.abspath(".venv/bin/python"),
                         os.path.abspath("tests/opt_functions.py"),
                         "--", "--name={}".format(f.name)]

            logging.info("init_args %s", init_args)
            # print(init_args)

            run_main(init_args)

            run_main(["run", *chdir, "--n_iter=12", "--n_parallel=3",
                      "--sleep=0.1"])

            from io import StringIO
            from contextlib import redirect_stdout
            f = StringIO()

            with redirect_stdout(f):
                run_main(["exp", *chdir, "-b"])

            best_result = float(f.getvalue())
            self.assertAlmostEqual(160.0, best_result, 2)
