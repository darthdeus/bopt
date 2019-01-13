default: bo-exp

watch:
	watch -n1 expstat results/rl-monte-carlo

bo-exp:
	PYTHONPATH=. python experiments/bo_rl_experiment.py

simple-exp:
	PYTHONPATH=. python experiments/simple_function_experiment.py

clean:
	rm -rf results/rl-monte-carlo

setup-venv:
	python -m virtualenv venv
	./venv/bin/pip install -r requirements.txt

plots:
	PYTHONPATH=. python misc/gp_images.py

rl-experiment:
	PYTHONPATH=. python experiments/basic_rl_experiment.py

benchmarks:
	PYTHONPATH=. python -m cProfile -s cumtime "benchmarks/$1.py" 2>&1 > "results/be-$1.txt"

mypy:
	mypy --ignore-missing-imports bopt

web:
	PYTHONPATH=. python app.py "results/simple-function"
