- nemuzu se ptat na vysledky samplu, aniz bych vedel adresar outputu jobu
  - jak a kdy mam prelejt outputy jobu do samplu? mam to vubec delat?



def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")
    import traceback
    try: raise
    except:
        logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
            '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
    return L





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
