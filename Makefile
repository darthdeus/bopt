default: rl-experiment

setup-venv:
	python -m virtualenv venv
	./venv/bin/pip install -r requirements.txt

plots:
	PYTHONPATH=. python misc/gp_images.py

rl-experiment:
	PYTHONPATH=. python experiments/basic_rl_experiment.py
