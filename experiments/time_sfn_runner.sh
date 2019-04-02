#!/bin/bash
set -ex

wd=$1
shift

time $wd/.venv/bin/python $wd/experiments/simple_function.py $@
