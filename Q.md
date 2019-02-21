- GPy normalizer ! ?
- store likelihoods
- last model pryc?
- yaml sort keys? (custom key order)



- co kdyz mam duplicitni-pid?
  - job-PID-1
  - pouzivat relativni cesty


<!-- - GPR failuje assert u solve -->
<!--  -->
<!-- - gpy vs bopt tests :O -->
<!--   - likelihoody vypadaji stejne ... multimodalni? -->
<!--   - mozna TF failuje, protoze to nebezi s restartama? -->
<!--     - kdyz necham jenom porovnani likelihoodu tak to funguje -->

- GPy !!!

- zamykani - flock, lockfile

- bopt cmdline
  - vytvoreni experimentu
  - pousteni jobu
    - single shot run
    - forever
      - lock
      - sync
      - run
      - unlock
      - sleep
  - suggest
    - naformatovany command s newlinama
  - bopt manual-run
  - bopt web
    - sync + render
  - bopt job -c DIR ID
  - bopt exp -c DIR



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






## oooooooooooooooooooooooooooooooooooooooooooold   |
##                                                  v






- [x] normalizace dat?

- [x] TF (LBFGS & SGD) fixed
  - [x] konverguje ke stejnym param jako LBFGS ze scikitu (prakticky identicke)

- [x] na 1d fitnu podobne ale ne identicke param jako GPy
  - [x] nll mam mensi

- [x] na 2d fitnu trochu jinak
  - [x] najdu      -3.4
  - [x] gpy       -24.0
  - [x] moje(gpy)  35.8

  - [x] z nejakeho duvodu vracim vetsi NLL pro stejne param a plotuju jinak

- [x] ale gpy samo sebe plotne stejne jako ja plotnu sebe (ale porad rika jiny nll)





- do rezu dat jenom max bod (muzem si ho vybrat)
  - cislo samplu
- vsechny body zobrazit v 2d viz
  - pca






- boundy v lbfgs
- merit MI mezi dimenzema?



- double fork pajp.py
- co vsechno chci ukladat do resultu
  - jak ukladam parametry?
- kde vezmu finish date?
- serialize/deserialize
- global opt hmc e^-f(x)?
- (n,) vs (n,1)
