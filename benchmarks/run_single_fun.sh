#!/bin/bash

VENV=${VENV:-~/.venv/tf-gpu}
REL=${BASH_SOURCE%/*}

PYTHONPATH="$REL/.." "$VENV/bin/python" "$REL/run_single_fun.py" $(sed "${SGE_TASK_ID}q;d" "$REL/args.txt")
