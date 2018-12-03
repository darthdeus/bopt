qsub -N optfun -o results/acqfn-100-bf -t 1-64 -v VENV=./venv -v SGE_TASK_ID=1 bash ./benchmarks/run_single_fun.sh
