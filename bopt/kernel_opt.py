import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from typing import Callable, Tuple
from numpy.linalg import inv, cholesky, det, solve
from scipy.optimize import minimize

from bopt.kernels import Kernel, SquaredExp

def kernel_log_likelihood(kernel: Kernel, X_train: np.ndarray,
                          y_train: np.ndarray, noise_level: float = 0) -> float:
    noise = noise_level ** 2 * np.eye(len(X_train))
    K = kernel(X_train, X_train) + noise

    # L = cholesky(K)
    # t1 = 0.5 * y_train.T @ solve(L.T, solve(L, y_train))

    t1 = 0.5 * y_train.T @ solve(K, y_train)

    # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html

    sign, logdet = np.linalg.slogdet(K)
    # t2 = 0.5 * 2 * np.sum(np.log(np.diagonal(cholesky(K))))
    t2 = logdet

    t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

    loglikelihood = t1 + t2 + t3

    # print(loglikelihood, kernel.l, kernel.sigma)

    # TODO: check this
    # assert loglikelihood >= 0, f"got negative log likelihood={loglikelihood}, t1={t1}, t2={t2}, t3={t3}"

    return loglikelihood


def tf_sqexp_kernel(a: tf.Tensor, b: tf.Tensor, ls: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    # TODO: round indexes

    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)

    sqnorm = stable_eye = tf.reduce_sum((a - b) ** 2.0, axis=2)

    exp = - (0.5 / tf.pow(ls, 2)) * sqnorm
    return tf.pow(sigma, 2) * tf.exp(exp)


def compute_optimized_kernel_tf(X_train, y_train, noise_level: float, kernel: Kernel = SquaredExp()) \
        -> Tuple[Kernel, float]:
    tf.enable_eager_execution()

    assert X_train.ndim == 2, X_train.ndim
    assert y_train.ndim == 1

    print(X_train.dtype)

    if not isinstance(kernel, SquaredExp):
        raise NotImplementedError()

    def_sigma, def_ls = kernel.default_params(X_train, y_train)

    noise = noise_level ** 2 * np.eye(len(X_train))

    sigma = tf.Variable(def_sigma, dtype=tf.float64)
    ls = tf.Variable(def_ls, dtype=tf.float64)

    global_step = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer()

    for i in range(1000):
        with tf.GradientTape() as tape:
            K = tf_sqexp_kernel(X_train, X_train, ls, sigma) + noise
            # K = kernel(X_train, X_train) + noise

            y_train_expanded = tf.expand_dims(y_train, 0)

            solved = tf.linalg.solve(K, tf.transpose(y_train_expanded))

            t1 = 0.5 * y_train_expanded @ solved
            # t1 = 0.5 * y_train.T @ solve(K, y_train)

            # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
            t2 = tf.linalg.slogdet(K)
            # t2 = tf.reduce_sum(tf.log(tf.linalg.diag(tf.linalg.cholesky(K))))
            # t2 = 0.5 * 2 * np.sum(np.log(np.diagonal(cholesky(K))))

            t3 = 0.5 * len(X_train) * np.log(2 * np.pi)

            nll = (t1 + t2 + t3)

        variables = [sigma, ls]
        grads = tape.gradient(nll, variables)

        optimizer.apply_gradients(zip(grads, variables), global_step=global_step)

        # if i % 20 == 0:
        #     print("{:3d}: {}".format(global_step.numpy(), nll.numpy()))

    return SquaredExp(l=ls.numpy().item(), sigma=sigma.numpy().item()), noise_level


def compute_optimized_kernel(kernel, X_train, y_train) -> Tuple[Kernel, float]:
    # TODO: remove & optimize noise
    noise_level = 0.492
    USE_TF = False
    # USE_TF = True

    # TODO:
    # assert X_train.dtype == y_train.dtype

    if USE_TF:
        X_train = X_train.reshape(-1, 1).astype(np.float64)
        y_train = y_train.astype(np.float64)
        return compute_optimized_kernel_tf(X_train, y_train, noise_level, kernel)
    else:
        def step(theta):
            nll = kernel_log_likelihood(kernel.set_params(theta[:-1]), X_train, y_train, theta[-1])
            # print(nll)
            return nll

        default_params = np.array(kernel.default_params(X_train, y_train).tolist() + [0.0])

        if len(default_params) == 0:
            return kernel

        bounds_with_noise = kernel.param_bounds() + [(1e-5, None)]

        res = minimize(step,
                       default_params,
                       bounds=bounds_with_noise,
                       method="L-BFGS-B",
                       tol=0,
                       options={"maxiter": 100})

        kernel.set_params(res.x)
        return kernel, res.x[-1]
