qsub -N optfun -o results -t 1-24 -v VENV=./venv -v SGE_TASK_ID=1 bash ./benchmarks/run_single_fun.sh
