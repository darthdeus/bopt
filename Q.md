GPSS.CC !!!!!
http://deepbayes.ru

# TODO: co kdyz dostanu manual evaluation, zkusit precejenom fitnout model
#       ale do plotu napsat, ze ten model neni podle ceho byl vybrany?

- plot current max
- fix vmin/vmax

- GPyOpt?

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
- vsechny body zobrazit v 2d viz
  - pca



## oooooooooooooooooooooooooooooooooooooooooooold   |
##                                                  v



- merit MI mezi dimenzema?



- double fork pajp.py
- kde vezmu finish date?
