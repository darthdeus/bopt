- TF (LBFGS & SGD) fixed
  - konverguje ke stejnym param jako LBFGS ze scikitu (prakticky identicke)

- na 1d fitnu podobne ale ne identicke param jako GPy
  - nll mam mensi

- na 2d fitnu trochu jinak
  - najdu      -3.4
  - gpy       -24.0
  - moje(gpy)  35.8

  - z nejakeho duvodu vracim vetsi NLL pro stejne param a plotuju jinak

- ale gpy samo sebe plotne stejne jako ja plotnu sebe (ale porad rika jiny nll)








- boundy v lbfgs
- merit MI mezi dimenzema?



- double fork pajp.py
- co vsechno chci ukladat do resultu
  - jak ukladam parametry?
- kde vezmu finish date?
- serialize/deserialize
- global opt hmc e^-f(x)?
- (n,) vs (n,1)
