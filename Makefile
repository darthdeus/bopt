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
