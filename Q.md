# TODO

- [x] skipnout prvnich num_random + 1 samplu u kernel param plotu
- moznost poustet jenom random search
- [x] timeline
- opravit zaokrouhlovani aby slo jen spravnymi smery :)
- fitnout mean fci
- ARD=True u kernelu
- [x] smazat mu_pred 0/1 u random search

- web
  - NxN grid mean + acq
    - 1d
    - 2d
  - interpolace mezi 2 body hyperparam + plot slicu v lib. 0d, 1d, 2d ose
  - burty!
  - okraje v plotech

  - graf acq fn
  - grafy v dalsim bode

- expected improvement per second

- zaokrouhlovani z nejakeho duvodu probiha spatne u intu

- [x] collect nejak dlouho trva
- [x] moznost nadefinovat qsub param (a nebo obecne spoustejici param co se pridaji u runneru)
- [x] logscale int je spatne, protoze dava jenom 2^n a zadne mezi?

- [x] logovat jak dlouho uz job bezi v bopt exp
- log timings

- [x] parsovat bash -c time pokud tam je, pokud ne tak beru finished_at jako ted
- [x] videt jak se meni hyperparam v case (s kazdym novym bodem + konvergence hyperparam)
  - porovnat s optmizer_restarts()
- [x] konvergencni graf


- [x] backupit yml predtim nez ho prepisuju
- [x] qstat jenom na joby co nemaji result

- gpy numpy error
- [x] vyresit, proc propose location vraci NAN
- zaokrouhlovaci kernel
  - diskretni hyperparam - onehot nebo fixni lengthscale na konkretni hyperparam
    - sigmoid vs round?
- [x] logscale

- bopt delete job_id
- bopt resubmit job_id

- seedy

- [x] job by mel byt optional
  - [x] manual sample nema job

  - [x] presunout run_params pod sample
  - [x] oznacit status WAITING_FOR_SIMILAR
  - [x] brat WAITING_FOR_SIMILAR v potaz behem bopt
  - [x] zprocessit WAITING_FOR_SIMILAR v collect_results

  - [x] manual_run - vyberu hyperparam rucne
  - [x] manual_sample - reknu kolik to vyslo "rucne"

- [x] zkontrolovat, ze se mu_pred + sigma_pred pouziva ve spravnych mistech
  - [x] u bezicich jobu bych mel brat mu_pred misto result (jeste ho nemam)

- [x] collect na vsech spravnych mistech
  - [x] nikde nepouzivat meta_dir (neni potreba, delame cd)

- [x] replace print with logging
- [x] garbage collecteni resultu
- [x] job status (sample status?)

- [x] regex parsujici result
- [x] collectit resulty do yamlu
  - [x] v ramci toho ziskat i finished_at

- [x] nepocitam s failnutyma jobama
  - [x] predpokladam, ze priste uspeje - tohle nedelam, nikdy nedelam retry, nepocitam s docasnyma chybama
  - [x] kontrolovat zaokrouhlene jestli ho nepoustim znova
    - [x] vytvorim manual sample s mean_pred - sigma_pred (nejak aby byl videt) a uz nepoustim dal

- [x] bopt debug - ipdb with bopt imports

- [x] davat mean kdyz delam znovu stejny bod
- [x] ignorovat underflow dokud nedelaj problem :)

- zdetekovat duplicitni ID a spadnout

- smazat benchmarks/ az bude cas udelat to poradne

-----------------------------------

- [x] manual run nefunguje
- [x] JOB_ID env variable - neni potreba, SGE to nastavi, u local to
  neresim
- [x] int range high

- [x] testovat vse na MNISTu a ne RL MC
- [x] u manual-run kontrolovat, ze hodnota je uvnitr range
  - [x] suggest vraci hodnoty mimo horni range

- [x] logging !!!!

- vsechny body zobrazit v 2d viz
  - pca

# Co chybi / je aktualne broken

- [x] kernel type:       trivialni, ale neni to :)
- [x] acq fn v yml
- [x] parallel evaluace: trivialni, ale neni to :)
  - [x] u nedobehlych jobu predpokladam ze vysledek je jejich mean
  - [x] moznost pustit run s -j 10
  - [x] kontrolovat, ze 2x nevyhodnocuju ve stejnym bode
- [x] intove hyperparam: GPy?
- [x] diskretni/categorical parametry
- [x] SGE Runner:        neni broken, jenom neni updated na novejsi API

- [x] plot convergence

# Pozdeji

- kernely! porad nevime ktery se nam libi
- acq fn   ... to same

# Asi vubec neresime

- paralelni evaluace: jde to lip nez pouzivat mean?
- duplicitni PIDy
- multijoby

- intermediate results !!!

# Co je od minule

- [x] GPy
- [x] cmdline: manual-run, run, run-single, plot, suggest
- [x] serializace modelu pro kazdy step
- [x] plotovani vsech kroku
- [x] konzistentni -C vsude

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


- [x] logging? ploty kdyz failne assert ... soft assert?


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

