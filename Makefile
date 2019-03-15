BOPT=./.venv/bin/python ./bin/bopt

default: sfntest

plot:
	rm -rf results/sfn/plots
	PYTHONPATH=. $(BOPT) plot -C results/sfn

test_serialization:
	pytest tests/test_todict_fromdict.py

test_serialization_ipdb:
	PYTHONPATH=. python tests/test_todict_fromdict.py

gpy_compare:
	PYTHONPATH=. python tests/test_gpy_comparison.py

sfntest:
	rm -rf results/sfn
	$(BOPT) init --param "x:int:0:5" --param "y:float:0:25" \
		--param "z:float:0:1" --param "w:float:2:7" \
		--param "activation:discrete:relu:sigmoid:tanh" \
		-C results/sfn \
		$(PWD)/.venv/bin/python $(PWD)/experiments/simple_function.py
	$(BOPT) run --n_iter=10 -C results/sfn
	# convert -delay 100 -loop 0 tmp/*.png anim.gif

sfntest2d:
	rm -rf results/sfn2d
	$(BOPT) init --param "x:float:0:5" --param "y:float:0:5" \
		-C results/sfn2d \
		$(PWD)/.venv/bin/python $(PWD)/experiments/simple_function2d.py
	$(BOPT) run -C results/sfn2d

mctest:
	rm -f tmp/*
	rm -rf results/mc
	$(BOPT) init --param "gamma:float:0:1" --param "epsilon:float:0:1" \
		-C results/mc \
		$(PWD)/.venv/bin/python $(PWD)/experiments/rl/monte_carlo.py
	$(BOPT) run -C results/mc
	# convert -delay 100 -loop 0 tmp/*.png anim.gif

init:
	$(BOPT) init --param "gamma:float:0:1" --param "epsilon:float:0:1" \
		-C results/mc \
		$(PWD)/.venv/bin/python $(PWD)/experiments/rl/monte_carlo.py

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

test:
	bash -c "source ./.venv/bin/activate && pytest && ./tests/test_cli.sh && make mypy"
