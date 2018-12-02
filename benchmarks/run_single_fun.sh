#!/bin/bash

# VENV=../venv
VENV=~/.venv/tf-gpu

PYTHONPATH=.. "$VENV/bin/python" run_single_fun.py $(sed "${SGE_TASK_ID}q;d" args.txt)
