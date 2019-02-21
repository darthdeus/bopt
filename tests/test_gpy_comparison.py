import tensorflow as tf
tf.enable_eager_execution()

import sys
import yaml
import unittest
import numpy as np
import math
import random

import GPy
from bopt import GaussianProcessRegressor, SquaredExp

from typing import NamedTuple, List, Tuple


# def param_nll_gpflow(X_train, y_train, params):
#     import gpflow
#     with gpflow.defer_build():
#         kernel = gpflow.kernels.RBF(
#                 input_dim=1,
#                 variance=params["sigma"] ** 2, # TODO !!!!!!!!!!!!! ** 2?
#                 lengthscales=params["ls"])
#
#
#         m = gpflow.models.GPR(X_train, y_train.reshape(-1, 1), kern=kernel)
#         m.likelihood.variance = params["noise"] ** 2
#
#     # __import__('ipdb').set_trace()


def param_nll_bopt(X_train, y_train, params):
    kernel = SquaredExp(l=params["ls"], sigma=params["sigma"])
    gp = GaussianProcessRegressor(noise=params["noise"], kernel=kernel).\
            fit(X_train, y_train)

    return gp.log_prob().numpy().item()


def param_nll_gpy(X_train, y_train, params):
    # kernel = SquaredExp(l=params["ls"], sigma=params["sigma"])
    # gp = GaussianProcessRegressor(noise=params["noise"], kernel=kernel).\
    #         fit(X_train, y_train)
    #
    # return gp.log_prob().numpy().item()
    rbf = GPy.kern.RBF(input_dim=1, variance=params["sigma"]**2, lengthscale=params["ls"])
    gpr = GPy.models.GPRegression(X_train, y_train.reshape(-1, 1), rbf)
    gpr.Gaussian_noise.variance = params["noise"] ** 2

    # gpr.optimize(max_iters=5000)

    # mu, cov = gpr.predict(X, full_cov=True)
    #
    # params = {
    #     "noise": float(np.sqrt(gpr.Gaussian_noise.variance)),
    #     "ls": gpr.rbf.lengthscale.values[0],
    #     "sigma": np.sqrt(gpr.rbf.variance.values[0])
    # }

    nll = -gpr.log_likelihood()

    return nll



def model_bopt(X_train, y_train, X):
    gp = GaussianProcessRegressor(noise=1.0, kernel=SquaredExp(l=1.0, sigma=1.0)).\
            fit(X_train, y_train).optimize_kernel().posterior(X)

    params = {
        "noise": gp.noise,
        "ls": gp.kernel.params["lengthscale"].numpy().item(),
        "sigma": gp.kernel.params["sigma"].numpy().item()
    }

    return gp.mu, gp.cov, params, gp.log_prob().numpy().item()


def model_gpy(X_train, y_train, X):
    rbf = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    gpr = GPy.models.GPRegression(X_train, y_train.reshape(-1, 1), rbf)

    gpr.optimize_restarts(max_iters=5000)

    mu, cov = gpr.predict(X, full_cov=True)

    params = {
        "noise": float(np.sqrt(gpr.Gaussian_noise.variance)),
        "ls": gpr.rbf.lengthscale.values[0],
        "sigma": np.sqrt(gpr.rbf.variance.values[0])
    }

    nll = -gpr.log_likelihood()

    return mu.reshape(-1), cov, params, nll


def generate_datasets() -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    def d1():
        X_train = np.array([2,2.5,20,100], dtype=np.float64).reshape(-1, 1)
        y_train = np.array([2,3,2,5], dtype=np.float64)

        X = np.arange(min(X_train) - 0.1, max(X_train) + 0.1, step=0.1).reshape(-1, 1)

        return X_train, y_train, X

    results = []

    results.append(d1())

    num_points = 20

    funs = [np.sin, np.cos, lambda x: x**2]

    for i in range(50):
        data_noise = random.random()
        X_train = np.random.random(num_points).astype(np.float64).reshape(-1, 1)

        y_train = random.choice(funs)(X_train) + np.random.random(1).astype(np.float64)
        y_train = y_train.reshape(-1)

        # y_train = np.random.random(num_points).astype(np.float64)

        X = np.arange(min(X_train) - 0.1, max(X_train) + 0.1, step=0.1).reshape(-1, 1)

        results.append((X_train, y_train, X))

    return results



class TestMatchingGPy(unittest.TestCase):
    def test_on_random_data(self):
        datasets = generate_datasets()

        with open("gen_datasets.yml", "w") as f:
            yaml.dump(datasets, f)

        for i, (X_train, y_train, X) in enumerate(datasets):
            print("\n\n")
            print("########### DATASET {} #######".format(i), file=sys.stderr)

            m1, c1, p1, nll1 = model_bopt(X_train, y_train, X)
            m2, c2, p2, nll2 = model_gpy(X_train, y_train, X)

            print()
            print("PARAMS")
            print(p1)
            print(p2)

            print("NLL")
            print(nll1)
            print(nll2)
            print()

            print("max mean δ:\t", np.max(np.abs(m1 - m2)))
            print("max cov  δ:\t", np.max(np.abs(c1 - c2)))
            print("max nll  δ:\t", np.max(np.abs(nll1 - nll2)))

            print()

            print("B -> B:\t", param_nll_bopt(X_train, y_train, p1))
            print("G -> B:\t", param_nll_bopt(X_train, y_train, p2))

            print("B -> G:\t", param_nll_gpy(X_train, y_train, p1))
            print("G -> G:\t", param_nll_gpy(X_train, y_train, p2))

            # param_nll_gpflow(X_train, y_train, p1)

            self.assertAlmostEqual(nll1, nll2, places=1)
            self.assertSetEqual(set(p1.keys()), set(p2.keys()))

            for key in p1.keys():
                self.assertAlmostEqual(p1[key], p2[key], delta=1e-1)

            self.assertTrue(np.allclose(c1, c2, atol=1e-3), np.allclose(m1, m2, atol=1e-3))


            # plt.plot(np.diag(c1 - c2))
            # plt.show()
            # plt.imshow(c1 - c2)
            # plt.show()
            #
            # plt.plot(m1 - m2)
# np.diag(c1 - c2)


if __name__ == "__main__":
    unittest.main()
