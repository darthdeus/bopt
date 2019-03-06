#!/bin/bash

set -ex

test_dir=tests/tmp/cli_test

rm -rf "$test_dir"
mkdir -p "$test_dir"

export PYTHONPATH=.

bopt init -C "$test_dir" --param "x:float:0:5" --param "y:float:0:25" \
  --param "z:float:0:1" --param "w:float:2:7" \
  ./.venv/bin/python ./experiments/simple_function.py

bopt run-single -C "$test_dir"

bopt plot -C "$test_dir"
bopt suggest -C "$test_dir"
bopt web -C "$test_dir" &
sleep 5
kill -9 %1

# TODO: co kdyz davam hodnotu mimo range?
# TODO: int !!!
bopt manual-run -C "$test_dir" --x=0.1 --y=0.3 --z=0.1 --w=4
bopt exp -C "$test_dir"

