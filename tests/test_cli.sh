#!/bin/bash

set -ex

test_dir=tests/tmp/cli_test

export PATH="./bin:$PATH"
export PYTHONPATH=.

bopt=./.venv/bin/bopt

rm -rf "$test_dir"
mkdir -p "$test_dir"

$bopt init -C "$test_dir" \
  --param "x:int:0:5" \
  --param "y:logscale_int:1:128" \
  --param "z:float:0:6" \
  --param "w:logscale_float:1:7" \
  --param "activation:discrete:relu:sigmoid:tanh" \
  --ard=1 --gamma-prior=1 --kernel=rbf --num-optimize-restarts=1 --fit-mean=0 \
  $PWD/.venv/bin/python $PWD/experiments/simple_function.py


$bopt run-single -C "$test_dir"
# sleep 5
$bopt run -C "$test_dir" --n_iter=4 --n_parallel=2

$bopt plot -C "$test_dir"
$bopt suggest -C "$test_dir"
$bopt web -C "$test_dir" &
sleep 3
kill -9 %1

# TODO: test out of bounds
$bopt exp -C "$test_dir"

$bopt debug -C "$test_dir" &
sleep 3
kill -9 %1

set +e

# TODO: check out tests/tmp ... fix ",." dir
if $bopt manual-run -C "$test_dir" \
  --x=10 --y=129 --z=1.0 --w=4.3 --activation=sigmoid; then
  echo "Param values outside should have failed"
  exit 1
fi

set -e

$bopt manual-run -C "$test_dir" \
  --x=4 --y=126 --z=1.0 --w=4.3 --activation=sigmoid
