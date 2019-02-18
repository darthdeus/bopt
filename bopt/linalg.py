import lapack
import numpy as np
import numpy.linalg as linalg
import logging


# TODO: missing source/ref
def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    diagA = np.diag(A)

    if np.any(diagA <= 0.):
        raise linalg.LinAlgError("not pd: non-positive diagonal elements")

    jitter = diagA.mean() * 1e-6
    num_tries = 1

    while num_tries <= maxtries and np.isfinite(jitter):
        try:
            L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
            if num_tries > 1:
                logging.info("solved jitchol with {} tries".format(num_tries))
            return L
        except:
            jitter *= 10
        finally:
            num_tries += 1
    raise linalg.LinAlgError("not positive definite, even with jitter.")

    # TODO: which one do I actually want?
    import traceback
    try: raise
    except:
        logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
            '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
    return L

