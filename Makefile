BOPT=./.venv/bin/bopt

a:
	cp meta2.yml meta.yml
	bopt run --n_iter=1

default: mypy

plot:
	rm -rf results/sfn/plots
	PYTHONPATH=. $(BOPT) plot -C results/sfn

test_serialization:
	pytest tests/test_todict_fromdict.py

test_serialization_ipdb:
	PYTHONPATH=. python tests/test_todict_fromdict.py

gpy_compare:
	PYTHONPATH=. python tests/test_gpy_comparison.py

		# --param "x:int:0:5" \
		# --param "z:float:0:6" \
		# --param "w:logscale_float:1:7" \
		# --param "activation:discrete:relu:sigmoid:tanh" \

sfntest:
	rm -rf results/sfn
	$(BOPT) init \
		--param "y:float:1:3" \
		--qsub=-q \
		--qsub=cpu-troja.q \
		--gamma-a=1.0 --gamma-b=0.001 \
		--informative-prior=1 \
		--manual-arg-fname=$(PWD)/experiments/foo.txt \
		--manual-arg-fname=$(PWD)/experiments/bar.txt \
		--acq-xi=0.01 \
		-C results/sfn \
		$(PWD)/experiments/time_sfn_runner.sh $(PWD)
	$(BOPT) run --n_iter=10 --n_parallel=1 --sleep=0.1 -C results/sfn

sfntest2d:
	rm -rf results/sfn2d
	$(BOPT) init \
		--param "x:float:-5:5" \
		--param "y:float:-5:5" \
		--num-optimize-restarts=1 \
		-C results/sfn2d \
		$(PWD)/.venv/bin/python $(PWD)/experiments/simple_function2d.py
	$(BOPT) run --n_iter=100 -C results/sfn2d

lab02-cartpole:
	rm -rf results/l2cartpole
	$(BOPT) init \
		--param "batch_size:int:2:100" \
		--param "epochs:int:1:100" \
		--param "layers:int:2:10" \
		--param "units:int:2:50" \
		-C results/l2cartpole \
		$(HOME)/projects/npfl114/labs/02/bopt_cartpole.sh
	# $(BOPT) run --n_iter=80 -C results/l2cartpole

sfntest-sge:
	rm -rf results/sfn
	$(BOPT) init --param "x:int:0:5" --param "y:float:0:25" \
		--param "z:float:0:1" --param "w:float:2:7" \
		--param "activation:discrete:relu:sigmoid:tanh" \
		--runner sge \
		-C results/sfn \
		$(PWD)/.venv/bin/python $(PWD)/experiments/simple_function.py
	# $(BOPT) run --n_iter=10 -C results/sfn

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
	./.venv/bin/mypy bopt

clean-dist:
	rm -rf bopt.egg-info/ dist/ build/

test:
	bash -c "source ./.venv/bin/activate && pytest && ./tests/test_cli.sh && make mypy"
