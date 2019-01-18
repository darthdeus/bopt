# Master Thesis [![Build Status](https://travis-ci.com/darthdeus/master-thesis-code.svg?token=9CyU7Xa9qUJ9aPDDUHrX&branch=master)](https://travis-ci.com/darthdeus/master-thesis-code)

## TODO:

- [ ] TFE optimizer for SqExp kernel
- [ ] fix travis
- [x] marginal & conditional plots
- [x] share z-axis in plots
- [ ] doublefork children so they don't need to be awaited & can survive crash of parent
- [ ] noise optimization

- [x] discrete hyperparameters
  - [ ] https://arxiv.org/pdf/1706.03673.pdf
- [ ] priors
- [x] parallel coords graph
- [x] outer loop for optimization
- [x] display last line in error in parser when it fails?
- [ ] UCB acquisition function
- [ ] do we need L-BFGS?
  - [ ] compare optimizer performance between L-BFGS & SGD with TF
- [ ] parallel optimization without cmdline target
- [ ] expected improvement per second (hyperparam affects training time)
- [ ] look into approximate GP inference ... at which point would we need it?
- [ ] predicting training curves
  - [ ] ability to stop a job when it looks like it won't work out

## Kernels (http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/):

- [ ] Linear
- [ ] Polynomial
- [x] Gaussian (RBF)
- [ ] Matern
  - at this point it is unclear if we need anything but n/2
- [ ] Exponential
- [ ] Laplacian
- [ ] ANOVA
- [ ] tanh (sigmoid)
- [ ] Rational Quadratic
- [ ] Multiquadratic
- [ ] Inverse Multiquadratic
- [ ] Circular
- [ ] Spherical
- [ ] Wave
- [ ] Power
- [ ] Log
- [ ] Spline
- [ ] B-Spline
- [ ] Bessel
- [ ] Cauchy
- [ ] Chi-Square
- [ ] Histogram intersection
- [ ] Generalized histogram intersection
- [ ] Generalized t-student
- [ ] Bayesian
- [ ] Wavelet

## Runner notes

```
- start_script(args)
- intermediate_results
- is_finished
- results

class Hyperparameter:
  name: str
  range: Range

class Experiment:
  def __init__(self,
    hyperparameters,
    runner)

  hyperparameters: List[Hyperparameter]
  runner: Runner
  evaluations: List[Job]


class Runner():
  def start(args) -> Job

class Job:
  def state():
    is_finished: Bool
    intermediate_results : List[(Timestamp, Value)]
    final_result: Optional[Value]

  def kill()

  def serialize()
  def deserialize()

class SGERunner():
  def __init__(self, script_path, output_type)
```
