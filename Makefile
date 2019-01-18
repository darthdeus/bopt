# default: bo-exp
default: simple-exp

watch:
	watch -n1 expstat results/rl-monte-carlo

bo-exp: rl-experiment

simple-exp:
	PYTHONPATH=. python experiments/simple_function_experiment.py

rl-experiment:
	PYTHONPATH=. python experiments/bo_rl_experiment.py


web:
	PYTHONPATH=. python app.py "results/simple-function"

clean:
	rm -rf results/rl-monte-carlo

setup-venv:
	python -m virtualenv venv
	./venv/bin/pip install -r requirements.txt

plots:
	PYTHONPATH=. python misc/gp_images.py

benchmarks:
	PYTHONPATH=. python -m cProfile -s cumtime "benchmarks/$1.py" 2>&1 > "results/be-$1.txt"

mypy:
	mypy --ignore-missing-imports bopt

clean-dist:
	rm -rf bopt.egg-info/ dist/ build/
