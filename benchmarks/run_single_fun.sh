#!/bin/bash

VENV=${VENV:-~/.venv/tf-gpu}
REL=${BASH_SOURCE%/*}

SCRIPT_ARGS=$(sed "${SGE_TASK_ID}q;d" "$REL/args.txt")

echo "Running $SGE_TASK_ID with '$SCRIPT_ARGS'"
PYTHONPATH="$REL/.." "$VENV/bin/python" "$REL/run_single_fun.py" $SCRIPT_ARGS
