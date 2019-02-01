import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


from typing import Callable, Tuple, List
from numpy.linalg import inv, cholesky, det, solve
from scipy.optimize import minimize

from bopt.kernels import Kernel, SquaredExp


def print_rounded(*args):
    print("\t".join(list(map(lambda x: str(round(x, 3).item()), args))))


def tf_kernel_nll(kernel: Kernel, X_train: np.ndarray, y_train: np.ndarray, noise_level):
    # K = kernel.kernel(X_train, X_train) + noise
    #
    # # L = cholesky(K)
    # # t1 = 0.5 * y_train.T @ solve(L.T, solve(L, y_train))
    #
    # t1 = 0.5 * y_train.T @ solve(K, y_train)
    #
    # # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
    #
    # sign, logdet = np.linalg.slogdet(K)
    # # t2 = 0.5 * 2 * np.sum(np.log(np.diagonal(cholesky(K))))
    # t2 = logdet
    #
    # t3 = 0.5 * len(X_train) * np.log(2 * np.pi)
    #
    # loglikelihood = t1 + t2 + t3
    #
    # # print_rounded(kernel.l, kernel.sigma, noise_level, loglikelihood)
    #
    # return loglikelihood

    noise = tf.eye(len(X_train), dtype=tf.float64) * noise_level**2

    K = tf_sqexp_kernel(X_train, X_train,
            kernel.params["lengthscale"], kernel.params["sigma"]) + noise

    t1 = tf.transpose(y_train) @ tf.linalg.solve(K, y_train)
    t2 = 2*tf.linalg.slogdet(K).log_abs_determinant

    # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
    # log(det(K)) = log(det(L' @ L)) = log(det(L') * det(L)) =
    # = 2*log(det(L)) = 2*log(prod(diag(L))) = 2*sum(log(diag(L)))

    # L = tf.cholesky(K)
    # t1 = tf.transpose(y_train_expanded) @ tf.linalg.cholesky_solve(L, y_train_expanded)
    # t2 = 2 * tf.reduce_sum(tf.log(tf.linalg.tensor_diag_part(L)))

    # trace.append((t2 - t2_g).numpy())

    t3 = len(X_train) * np.log(2 * np.pi).item()

    nll = 0.5 * tf.squeeze(t1 + t2 + t3)

    # print_rounded(ls.numpy(), sigma.numpy(), noise_level.numpy(), nll.numpy())
    return nll



def kernel_log_likelihood(kernel: Kernel, X_train: np.ndarray,
                          y_train: np.ndarray, noise_level: float = 0) -> float:
    noise = noise_level ** 2 * np.eye(len(X_train))
    K = kernel(X_train, X_train) + noise

    # L = cholesky(K)
    # t1 = 0.5 * y_train.T @ solve(L.T, solve(L, y_train))

    t1 = y_train.T @ solve(K, y_train)

    # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html

    sign, logdet = np.linalg.slogdet(K)
    t2 = 2*logdet
    # t2 = 2 * np.sum(np.log(np.diagonal(cholesky(K))))

    t3 = len(X_train) * np.log(2 * np.pi)

    loglikelihood = 0.5 * (t1 + t2 + t3)

    # print_rounded(kernel.l, kernel.sigma, noise_level, loglikelihood)

    return loglikelihood


def tf_sqexp_kernel(a: tf.Tensor, b: tf.Tensor, ls: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    # TODO: round indexes

    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)

    sqnorm = stable_eye = tf.reduce_sum((a - b) ** 2.0, axis=2)

    exp = - (0.5 / tf.pow(ls, 2)) * sqnorm
    return tf.pow(sigma, 2) * tf.exp(exp)


def compute_optimized_kernel_tf(X_train, y_train, kernel: Kernel) \
        -> Tuple[Kernel, float]:
    assert X_train.ndim == 2, X_train.ndim
    assert y_train.ndim == 2

    if not isinstance(kernel, SquaredExp):
        raise NotImplementedError()

    def_sigma, def_ls = kernel.default_params(X_train, y_train)
    def_noise = 0.0

    trace: List[float] = []

    bounds_fn_tf = lambda x: tf.nn.softplus(x) + 1e-5
    bounds_fn_np = lambda x: np.logaddexp(0, x) + 1e-5

    def tf_nll(ls_var, sigma_var, noise_level_var):
        with tf.GradientTape() as tape:
            tape.watch(ls_var)
            tape.watch(sigma_var)
            tape.watch(noise_level_var)

            ls          = bounds_fn_tf(ls_var)
            sigma       = bounds_fn_tf(sigma_var)
            noise_level = bounds_fn_tf(noise_level_var)


            noise = tf.eye(len(X_train), dtype=tf.float64) * noise_level**2

            K = tf_sqexp_kernel(X_train, X_train, ls, sigma) + noise

            t1 = tf.transpose(y_train) @ tf.linalg.solve(K, y_train)
            t2 = 2*tf.linalg.slogdet(K).log_abs_determinant

            # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
            # log(det(K)) = log(det(L' @ L)) = log(det(L') * det(L)) =
            # = 2*log(det(L)) = 2*log(prod(diag(L))) = 2*sum(log(diag(L)))

            # L = tf.cholesky(K)
            # t1 = tf.transpose(y_train_expanded) @ tf.linalg.cholesky_solve(L, y_train_expanded)
            # t2 = 2 * tf.reduce_sum(tf.log(tf.linalg.tensor_diag_part(L)))

            # trace.append((t2 - t2_g).numpy())

            t3 = len(X_train) * np.log(2 * np.pi)

            nll = 0.5 * tf.squeeze(t1 + t2 + t3)

            # print(ls.numpy(), sigma.numpy(), noise_level.numpy(), nll.numpy())

            trace.append(nll)
            assert nll.ndim == 0, f"got {nll.ndim} with shape {nll.shape}"

        return nll, tape

    def value_and_gradients(params):
        params = tf.cast(params, tf.float64)

        nll, tape = tf_nll(*params)

        grads = tape.gradient(nll, params)

        grads_ = tf.constant(list(map(lambda x: x.numpy(), grads)))

        return nll, grads_

    USE_LBFGS = True
    USE_LBFGS = False

    def optimize_bfgs():
        import tensorflow_probability as tfp

        init = tf.constant([def_ls, def_sigma, def_noise], dtype=tf.float64)
        result = tfp.optimizer.bfgs_minimize(
            value_and_gradients,
            initial_position=init,
            tolerance=tf.constant(1e-3, dtype=tf.float64)
        )

        return bounds_fn_np(result.position.numpy())

    def optimize_sgd():
        ls_var          = tf.Variable(def_ls,    dtype=tf.float64)
        sigma_var       = tf.Variable(def_sigma, dtype=tf.float64)
        noise_level_var = tf.Variable(def_noise, dtype=tf.float64)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=1e-3)
        variables = [ls_var, sigma_var, noise_level_var]

        for i in range(400):
            nll, tape = tf_nll(*variables)
            grads = tape.gradient(nll, variables)

            optimizer.apply_gradients(zip(grads, variables))

        return bounds_fn_np(list(map(lambda x: x.numpy(), variables)))


    PLOT_TRACE = True
    PLOT_TRACE = False

    if PLOT_TRACE:
        import matplotlib.pyplot as plt
        print(optimize_bfgs())
        plt.plot(trace)
        plt.show()
        trace = []

        print(optimize_sgd())
        plt.plot(trace)
        plt.show()
        trace = []

    if USE_LBFGS:
        ls, sigma, noise_level = optimize_bfgs()
    else:
        ls, sigma, noise_level = optimize_sgd()

    if PLOT_TRACE:
        plt.plot(trace)
        plt.show()

    return SquaredExp(l=ls, sigma=sigma), noise_level


def tf_kernel_nll_with_grads(X_train, y_train, noise_level_: float, kernel: Kernel, compute_grads: bool) \
        -> Tuple[Kernel, float]:
    assert X_train.ndim == 2, X_train.ndim
    assert y_train.ndim == 2

    if not isinstance(kernel, SquaredExp):
        raise NotImplementedError()

    noise_level = tf.constant(noise_level_, dtype=tf.float64)

    with tf.GradientTape() as tape:
        tape.watch(noise_level)

        nll = tf_kernel_nll(kernel, X_train, y_train, noise_level)

        assert nll.ndim == 0, f"got {nll.ndim} with shape {nll.shape}"

    if compute_grads:
        grad_vars = [*kernel.params.values(), noise_level]
        grads = tape.gradient(nll, grad_vars)
    else:
        grads = None

    return nll, grads


def compute_optimized_kernel(kernel, X_train, y_train) -> Tuple[Kernel, float]:
    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, -1)

    assert y_train.ndim == 2
    assert X_train.ndim == 2

    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.float64)

    # k, n = compute_optimized_kernel_tf(X_train, y_train, noise_level, kernel)

    # TODO: noise 1000 overflow
    USE_TF = False
    # USE_TF = True

    if USE_TF:
        return compute_optimized_kernel_tf(X_train, y_train, kernel)
    else:
        trace = []
        global i
        i = 0
        def step(theta):
            nll2 = kernel_log_likelihood(kernel.set_params(theta[:-1]), X_train, y_train, theta[-1])
            nll, grads = tf_kernel_nll_with_grads(X_train, y_train,
                                   theta[-1], kernel.set_params(theta[:-1]),
                                   compute_grads=False)

            nll = nll.numpy()


            trace.append(nll)
            global i

            if i % 20 == 0:
                print((nll - nll2), grads)
                # print(theta, nll)

            i += 1
            return nll

        default_params = np.array(kernel.default_params(X_train, y_train).tolist() + [0.0])
        # default_params = np.array([k.l, k.sigma, n])

        if len(default_params) == 0:
            return kernel

        bounds_with_noise = kernel.param_bounds() + [(1e-5, None)]

        res = minimize(step,
                       default_params,
                       bounds=bounds_with_noise,
                       method="L-BFGS-B",
                       tol=0,
                       options={"maxiter": 100})

        PLOT_TRACE = True
        PLOT_TRACE = False

        if PLOT_TRACE:
            import matplotlib.pyplot as plt
            plt.plot(trace)
            plt.show()

        kernel.set_params(res.x)
        return kernel, res.x[-1]
