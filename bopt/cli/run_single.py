import os
import sys
import yaml

import bopt
from bopt.cli.util import handle_cd, ensure_meta_yml

def run(args) -> None:
    handle_cd(args)

    with ensure_meta_yml():
        experiment = bopt.Experiment.deserialize(".")

        experiment.run_single(bopt.ModelConfig(args), ".,")
