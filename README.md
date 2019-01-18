# Bayesian Optimization of HyperParameters - `bopt` [![Build Status](https://travis-ci.com/darthdeus/master-thesis-code.svg?token=9CyU7Xa9qUJ9aPDDUHrX&branch=master)](https://travis-ci.com/darthdeus/master-thesis-code)

Available commands:

```python
# Run a Flask web interface to display experiment reuslts.
python -m bopt.web META_DIR

# Get an overview status of an experiment.
python -m bopt.expstat META_DIR

# Check the status of a job based on its ID.
python -m bopt.jobstat JOB_ID
```

# Installation

Since bopt depends on TensorFlow it is a good idea to run it either in Docker
or in a Python virtual environment. Both methods are equivalent and there is no
downside to choosing either.

## Virtualenv install

TODO

## Docker install

TODO
