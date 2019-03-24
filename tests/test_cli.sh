#!/bin/bash

set -ex

test_dir=tests/tmp/cli_test

rm -rf "$test_dir"
mkdir -p "$test_dir"

export PATH="./bin:$PATH"
export PYTHONPATH=.

bopt init -C "$test_dir" --param "x:float:0:5" --param "y:float:0:25" \
  --param "z:int:0:2" --param "w:discrete:relu:sigmoid" \
  $PWD/.venv/bin/python $PWD/experiments/simple_function.py

bopt run-single -C "$test_dir"
# sleep 5
bopt run -C "$test_dir" --n_iter=2 --n_parallel=2

bopt plot -C "$test_dir"
bopt suggest -C "$test_dir"
bopt web -C "$test_dir" &
sleep 3
kill -9 %1

# TODO: check out tests/tmp ... fix ",." dir
bopt manual-run -C "$test_dir" --x=0.1 --y=0.3 --z=1 --w=sigmoid
# TODO: test out of bounds
bopt exp -C "$test_dir"

bopt debug -C "$test_dir" &
sleep 3
kill -9 %1

# TODO: overit ze se vola vsechno
