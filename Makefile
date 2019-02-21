# default: bo-exp
# default: simple-exp

default: mctest

test_serialization:
	pytest tests/test_todict_fromdict.py

test_serialization_ipdb:
	PYTHONPATH=. python tests/test_todict_fromdict.py

gpy_compare:
	PYTHONPATH=. python tests/test_gpy_comparison.py

mctest:
	rm -f tmp/*
	rm -rf results/mc
	bopt init --param "gamma:float:0:1" --param "epsilon:float:0:1" --dir results/mc ./.venv/bin/python ./experiments/rl/monte_carlo.py
	bopt run results/mc
	convert -delay 60 -loop 0 tmp/*.png anim.gif

init:
	bopt init --param "gamma:float:0:1" --param "epsilon:float:0:1" --dir results/mc ./.venv/bin/python ./experiments/rl/monte_carlo.py

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
