- sigmoid vs round?
- interpolace mezi 2 body hyperparam + plot slicu v lib. 0d, 1d, 2d ose
- burty!
- okraje v plotech

- vsechny body zobrazit v 2d viz
  - pca

# Co chybi / je aktualne broken

- [x] kernel type:       trivialni, ale neni to :)
- acq fn v yml
- intove hyperparam: GPy?
- SGE Runner:        neni broken, jenom neni updated na novejsi API
- parallel evaluace: trivialni, ale neni to :)
  - u nedobehlych jobu predpokladam ze vysledek je jejich mean
  - kontrolovat, ze 2x nevyhodnocuju ve stejnym bode
  - jde to lip nez pouzivat mean?

- kernely! porad nevime ktery se nam libi
- acq fn   ... to same

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
