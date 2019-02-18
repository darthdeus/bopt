import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Callable, Tuple, List, Dict
from numpy.linalg import inv, cholesky, det, solve
from scipy.optimize import minimize

from bopt.kernels.kernels import Kernel, SquaredExp


os.environ["USE_LBFGS"] = "1"
os.environ["USE_LBFGS"] = "0"
os.environ["USE_TF"] = "1"
os.environ["USE_TF"] = "0"


def is_tensor(x):
    return isinstance(x, (tf.Tensor, tf.SparseTensor, tf.Variable))

PRINT_EACH = 50
PRINT_ITER = 0

def print_rounded(*args):
    print("    ".join(list(map(lambda x: "{:.6f}".format(x.item()), args))))


def tf_kernel_nll(X_train: np.ndarray, y_train: np.ndarray, ls, sigma, noise):
    assert is_tensor(ls)
    assert is_tensor(sigma)
    assert is_tensor(noise)

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

    # print("\n\n\n")
    # print_rounded(*map(lambda x: x.numpy(), [ls, sigma, noise]))
    # print("\n\n\n")

    # TODO: fuj, pryc s timhle a fixnout to poradne
    if y_train.ndim == 1:
        y_train = tf.expand_dims(y_train, -1)

    noise_mat = tf.eye(len(X_train), dtype=tf.float64) * noise**2

    K = tf_sqexp_kernel(X_train, X_train, ls, sigma) + noise_mat

    assert is_tensor(K)

    # print(np.diag(K.numpy()).mean(), K.numpy().mean())
    Ky = K.numpy()
    # print(np.diag(Ky).mean(), (Ky - np.diag(np.diag(Ky))).mean())

    t1 = tf.transpose(y_train) @ tf.linalg.solve(K, y_train)
    t2 = tf.linalg.slogdet(K).log_abs_determinant

    # https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
    # log(det(K)) = log(det(L' @ L)) = log(det(L') * det(L)) =
    # = 2*log(det(L)) = 2*log(prod(diag(L))) = 2*sum(log(diag(L)))

    # L = tf.cholesky(K)
    # t1 = tf.transpose(y_train_expanded) @ tf.linalg.cholesky_solve(L, y_train_expanded)
    # t2 = 2 * tf.reduce_sum(tf.log(tf.linalg.tensor_diag_part(L)))

    # trace.append((t2 - t2_g).numpy())

    t3 = len(X_train) * np.log(2 * np.pi).item()

    nll = 0.5 * tf.squeeze(t1 + t2 + t3)

    if not "nll" in param_traces:
        param_traces["nll"] = []

    param_traces["ls"].append(float(ls.numpy()))
    param_traces["sigma"].append(float(sigma.numpy()))
    param_traces["noise"].append(float(noise.numpy()))
    param_traces["nll"].append(float(nll.numpy()))

    # TODO: re-eneable
    global PRINT_ITER
    PRINT_ITER += 1
    if PRINT_ITER % PRINT_EACH == 0:
        print_rounded(ls.numpy(), sigma.numpy(), noise.numpy(), nll.numpy())


    return nll



def kernel_log_likelihood(kernel: Kernel, X_train: np.ndarray,
                          y_train: np.ndarray, noise_level: float = 0) -> float:
    ls = kernel.params["lengthscale"]
    sigma = kernel.params["sigma"]
    assert is_tensor(ls)
    assert is_tensor(sigma)

    noise = tf.Variable(noise_level, dtype=tf.float64)

    if False:
        noise = noise_level ** 2 * np.eye(len(X_train))
        K = kernel(X_train, X_train) + noise
        t1 = y_train.T @ solve(K, y_train)

        sign, logdet = np.linalg.slogdet(K)
        t2 = 2*logdet
        t3 = len(X_train) * np.log(2 * np.pi)

        return 0.5 * (t1 + t2 + t3)
    else:
        return tf_kernel_nll(X_train, y_train, ls, sigma, noise)


def tf_sqexp_kernel(a: tf.Tensor, b: tf.Tensor, ls: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    # TODO: round indexes

    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)

    sqnorm = stable_eye = tf.reduce_sum((a - b) ** 2.0, axis=2)

    # TODO: fuj fixnout poradne
    sqnorm = tf.cast(sqnorm, tf.float64)

    exp = - (0.5 / tf.pow(ls, 2)) * sqnorm
    return tf.pow(sigma, 2) * tf.exp(exp)


param_traces: Dict = {
    "ls": [],
    "sigma": [],
    "noise": [],
    "nll": []
}

def get_param_traces():
    return param_traces

def clear_param_traces():
    param_traces["ls"] = []
    param_traces["sigma"] = []
    param_traces["noise"] = []
    param_traces["nll"] = []


clear_param_traces()


def compute_optimized_kernel_tf(X_train, y_train, kernel: Kernel) \
        -> Tuple[Kernel, float]:
    assert X_train.ndim == 2, X_train.ndim
    assert y_train.ndim == 2

    # if not isinstance(kernel, SquaredExp):
    #     raise NotImplementedError()

    def_sigma, def_ls = kernel.default_params(X_train, y_train)
    def_noise = 0.0

    trace: List[float] = []

    bounds_fn_tf = lambda x: tf.nn.softplus(x) + 1e-5
    bounds_fn_np = lambda x: np.logaddexp(0, x) + 1e-5

    def value_and_gradients(params):
        ls, sigma, noise = tf.cast(params, tf.float64)

        ls_var    = tf.Variable(ls,    dtype=tf.float64)
        sigma_var = tf.Variable(sigma, dtype=tf.float64)
        noise_var = tf.Variable(noise, dtype=tf.float64)

        nll, grads = tf_kernel_nll_with_grads(
                X_train, y_train, ls_var, sigma_var, noise_var,
                bounds_fn_tf)

        grads_ = tf.stack(grads)

        return nll, grads_

    def optimize_bfgs():
        # ls: 0.4692210017785583
        # sigma: 149.12150786283382
        # noise: 34.913024881555735

        init = tf.constant([0.469, 150, 34], dtype=tf.float64)
        init = tf.constant([def_ls, def_sigma, def_noise], dtype=tf.float64)

        result = tfp.optimizer.bfgs_minimize(
            value_and_gradients,
            initial_position=init,
            parallel_iterations=8,
            max_iterations=200,
        )

        # print("#######################")
        # print("       LBFGS DONE      ")
        # print("#######################")

        return bounds_fn_tf(result.position).numpy()

    def optimize_sgd():
        init = tf.constant([0.469, 150, 34], dtype=tf.float64)
        init = tf.constant([def_ls, def_sigma, def_noise], dtype=tf.float64)
        # init = tf.constant([10, 100, 100], dtype=tf.float64)

        ls_var          = tf.Variable(init[0], dtype=tf.float64)
        sigma_var       = tf.Variable(init[1], dtype=tf.float64)
        noise_level_var = tf.Variable(init[2], dtype=tf.float64)

        global_step = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(1e-2, global_step, 300, 0.5)
        optimizer = tf.train.AdamOptimizer(1e-2)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=1e-3,
        #         global_step=global_step)

        variables = [ls_var, sigma_var, noise_level_var]

        for i in range(2000):
            global_step.assign_add(1)
            nll, grads = tf_kernel_nll_with_grads(
                    X_train, y_train,
                    ls_var, sigma_var, noise_level_var,
                    bounds_fn_tf)

            optimizer.apply_gradients(zip(grads, variables))

        # print("#######################")
        # print("         SGD DONE      ")
        # print("#######################")

        return bounds_fn_tf(tf.stack(variables)).numpy()


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

    if os.environ.get("USE_LBFGS", "0") == "1":
        ls, sigma, noise_level = optimize_bfgs()
    else:
        ls, sigma, noise_level = optimize_sgd()

    if PLOT_TRACE:
        plt.plot(trace)
        plt.show()

    return SquaredExp(l=ls, sigma=sigma), noise_level


def tf_kernel_nll_with_grads(X_train, y_train, ls, sigma, noise,
        constraint_transform: Callable, compute_grads: bool = True) \
        -> Tuple[Kernel, float]:
    assert X_train.ndim == 2, X_train.ndim
    assert y_train.ndim == 2

    assert isinstance(ls, tf.Variable)
    assert isinstance(sigma, tf.Variable)
    assert isinstance(noise, tf.Variable)

    with tf.GradientTape() as tape:
        # tape.watch(ls)
        # tape.watch(sigma)
        # tape.watch(noise)
        ls_ = constraint_transform(ls)
        sigma_ = constraint_transform(sigma)
        noise_ = constraint_transform(noise)

        nll = tf_kernel_nll(X_train, y_train, ls_, sigma_, noise_)

        assert nll.ndim == 0, f"got {nll.ndim} with shape {nll.shape}"

    if compute_grads:
        grad_vars = [ls, sigma, noise]
        grads = tape.gradient(nll, grad_vars)
    else:
        grads = None

    return nll, grads

global i
i = 0

def compute_optimized_kernel(kernel, X_train, y_train) -> Tuple[Kernel, float]:
    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, -1)

    assert y_train.ndim == 2
    assert X_train.ndim == 2

    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.float64)

    # k, n = compute_optimized_kernel_tf(X_train, y_train, noise_level, kernel)

    # TODO: noise 1000 overflow

    constraint_transform = lambda x: tf.nn.softplus(x) + 1e-5

    if os.environ.get("USE_TF", "0") == "1":
        return compute_optimized_kernel_tf(X_train, y_train, kernel)
    else:
        global i
        i = 0
        def step(theta):
            # nll2 = kernel_log_likelihood(kernel.set_params(theta[:-1]), X_train, y_train, theta[-1])

            ls_var = tf.Variable(theta[0], dtype=tf.float64)
            sigma_var = tf.Variable(theta[1], dtype=tf.float64)
            noise_var = tf.Variable(theta[2], dtype=tf.float64)

            # TODO: handle constraints manually?
            nll, grads = tf_kernel_nll_with_grads(X_train, y_train,
                    ls_var, sigma_var, noise_var, compute_grads=False,
                    constraint_transform=constraint_transform)

            nll = nll.numpy()

            global i

            # if i % 20 == 0:
            #     print((nll - nll2), grads)
                # print(theta, nll)

            i += 1
            return nll

        default_params = np.array(kernel.default_params(X_train, y_train).tolist() + [0.0])

        if len(default_params) == 0:
            return kernel

        res = minimize(step,
                       default_params,
                       method="L-BFGS-B",
                       tol=1e-6,
                       options={"maxiter": 100})

        transformed_x = constraint_transform(res.x).numpy()
        kernel.set_params(transformed_x)
        return kernel, transformed_x[-1]
