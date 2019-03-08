- testovat vse na MNISTu a ne RL MC


- sigmoid vs round?
- interpolace mezi 2 body hyperparam + plot slicu v lib. 0d, 1d, 2d ose
- burty!
- okraje v plotech

- vsechny body zobrazit v 2d viz
  - pca

# Co chybi / je aktualne broken

- [x] kernel type:       trivialni, ale neni to :)
- [x] acq fn v yml
- parallel evaluace: trivialni, ale neni to :)
  - u nedobehlych jobu predpokladam ze vysledek je jejich mean
  - kontrolovat, ze 2x nevyhodnocuju ve stejnym bode
  - jde to lip nez pouzivat mean?
- intove hyperparam: GPy?
- SGE Runner:        neni broken, jenom neni updated na novejsi API

- plot convergence

# Pozdeji

- kernely! porad nevime ktery se nam libi
- acq fn   ... to same

# Asi vubec neresime

- web:               neni aktualizovano pro GPy
- duplicitni PIDy + flock
- multijoby

- intermediate results !!!

# Co je od minule

- GPy
- cmdline: manual-run, run, run-single, plot, suggest
- serializace modelu pro kazdy step
- plotovani vsech kroku
- konzistentni -C vsude

# Nejasne

- kam generovat slice v plotu u EI a u 1d/2d?
  - asi option pro `bopt plot`?
  - nebo generovat hodne slicu? viz jupyter

- GPy priory (jupyter notebook)
  - chceme prior nebo jenom bound?
  - https://github.com/SheffieldML/GPy/issues/735

- GPyOpt? https://github.com/SheffieldML/GPyOpt
  - asi spis jenom vykrast


-----------------------------------------------

GPSS.CC
http://deepbayes.ru

-----------------------------------------------
-----------------------------------------------
-----------------------------------------------

# TODO: co kdyz dostanu manual evaluation, zkusit precejenom fitnout model
#       ale do plotu napsat, ze ten model neni podle ceho byl vybrany?

- plot current max
- fix vmin/vmax

- plot STD?
- yaml sort keys? (custom key order)



- co kdyz mam duplicitni-pid?
  - job-PID-1
  - pouzivat relativni cesty


- zamykani - flock, lockfile

- bopt cmdline
  - vytvoreni experimentu
  - pousteni jobu
    - single shot run
    - forever
      - lock
      - sync
      - run
      - sync
      - unlock
      - sleep
  - suggest
    - naformatovany command s newlinama
      bopt manual-run --x=1 --y=3
  - bopt manual-run
  - bopt web
    - sync + render
  - bopt job -c DIR ID
  - bopt exp -c DIR
  - bopt plot -c DIR



- nemuzu se ptat na vysledky samplu, aniz bych vedel adresar outputu jobu
  - jak a kdy mam prelejt outputy jobu do samplu? mam to vubec delat?

- ukladani meta_dir? ted ho vsude musim predavat, ale
  kdyz ho budu serializovat, tak pak nic nejde presunout

- kam patri result parser? viz kradeni stdout

- jak u multijobu poustet/schedulovat vic behu?

- format cmdline argu u init? vs template.yml
    bopt init results/mc ./.venv/bin/python ./experiments/rl/monte_carlo.py

    bopt init -p parser neur
    bopt init -p file_parser[fname] neur

    mkfifo p
    neur 2>p | parser
    parser < p


    bopt init --param "gamma:float:0:1" --param "epsilon:float:0:1" --dir results/mc ./.venv/bin/python ./experiments/rl/monte_carlo.py


- logging? ploty kdyz failne assert ... soft assert?


- do rezu dat jenom max bod (muzem si ho vybrat)
  - cislo samplu



## oooooooooooooooooooooooooooooooooooooooooooold   |
##                                                  v


- merit MI mezi dimenzema?

- double fork pajp.py
- kde vezmu finish date?



## oooooooooooooooooooooooooooooooooooooooooooold   |
## oooooooooooooooooooooooooooooooooooooooooooold   |
## oooooooooooooooooooooooooooooooooooooooooooold   |
##                                                  v
##                                                  v
##                                                  v



- flake8 + black

- experiment
  - hyperparam
  - runner
  - samples (noise?)
    - param + vysledek
    - per-sample noise?
    - model
      - kernel

      - random search
      - gp

    - job
      - vypocet
      - result parser
  - last model

- acq funce donstane optimizer
  - max f(posterior(R|data))

- grafy pro param kernelu

- [ ] levenberg marquardt

- test x^2

- [x] marginal & conditional plots
- [x] share z-axis in plots
- [ ] doublefork children so they don't need to be awaited & can survive crash of parent
- [x] noise optimization
- [ ] time slider

- [x] discrete hyperparameters
  - [ ] https://arxiv.org/pdf/1706.03673.pdf
- [ ] priors
- [ ] UCB acquisition function
- [ ] parallel optimization without cmdline target
- [ ] expected improvement per second (hyperparam affects training time)
- [ ] look into approximate GP inference ... at which point would we need it?
- [ ] predicting training curves
  - [ ] ability to stop a job when it looks like it won't work out

## Posledni konzultace:

- zobrazovat spravne max
- hodnoty hyperparam videt v bodech ve slicech
- zobrazit zafixovane param u kazdeho slicu
- zobrazit param nejlepsiho bodu
- zobrazovat body co se aktualne vyhodnocuji
  - hodnotu z GP (a jak to dopadne?)

- slider na celou vizualizaci
- timestampovat zacatky a konce jobu

- multijoby
  - zvlast megajob a single

- seedy pro joby
  - jmeno parametru pres ktery se predava seed

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
