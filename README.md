# Bayesian Optimization of HyperParameters - `bopt` [![Build Status](https://travis-ci.com/darthdeus/master-thesis-code.svg?token=9CyU7Xa9qUJ9aPDDUHrX&branch=master)](https://travis-ci.com/darthdeus/master-thesis-code)

Available commands:

```python
# Create a new experiment
bopt new META_DIR

# Start tuning hyperparameters
bopt run META_DIR

# Run a Flask web interface to display experiment reuslts.
bopt web META_DIR

# Get an overview status of an experiment.
bopt exp META_DIR

# Check the status of a job based on its ID.
bopt job JOB_ID
```

# Installation

Since bopt depends on TensorFlow it is a good idea to run it either in
Docker or in a Python virtual environment. Both methods are equivalent and
there is no downside to choosing either.

## Virtualenv install

TODO

## Docker install

TODO
