BOPT=./.venv/bin/python ./bin/bopt

default: plot

plot:
	rm -rf results/sfn/plots
	PYTHONPATH=. $(BOPT) plot results/sfn

test_serialization:
	pytest tests/test_todict_fromdict.py

test_serialization_ipdb:
	PYTHONPATH=. python tests/test_todict_fromdict.py

gpy_compare:
	PYTHONPATH=. python tests/test_gpy_comparison.py

sfntest:
	rm -rf results/sfn
	$(BOPT) init --param "x:float:0:5" --param "y:float:0:25" \
		--param "z:float:0:1" --param "w:float:2:7" \
		-C results/sfn \
		./.venv/bin/python ./experiments/simple_function.py
	$(BOPT) run results/sfn
	# convert -delay 100 -loop 0 tmp/*.png anim.gif

mctest:
	rm -f tmp/*
	rm -rf results/mc
	$(BOPT) init --param "gamma:float:0:1" --param "epsilon:float:0:1" -C results/mc ./.venv/bin/python ./experiments/rl/monte_carlo.py
	$(BOPT) run results/mc
	# convert -delay 100 -loop 0 tmp/*.png anim.gif

init:
	$(BOPT) init --param "gamma:float:0:1" --param "epsilon:float:0:1" -C results/mc ./.venv/bin/python ./experiments/rl/monte_carlo.py

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
