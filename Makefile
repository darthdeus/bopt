default: rl-experiment

watch:
	watch -n1 expstat results/rl-monte-carlo

bo-exp:
	PYTHONPATH=. python experiments/bo_rl_experiment.py

setup-venv:
	python -m virtualenv venv
	./venv/bin/pip install -r requirements.txt

plots:
	PYTHONPATH=. python misc/gp_images.py

rl-experiment:
	PYTHONPATH=. python experiments/basic_rl_experiment.py

benchmarks:
	PYTHONPATH=. python -m cProfile -s cumtime "benchmarks/$1.py" 2>&1 > "results/be-$1.txt"
