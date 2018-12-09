import matplotlib.pyplot as plt
import numpy as np

from myopt.kernel import Matern, SquaredExp, RationalQuadratic
from myopt.gaussian_process import GaussianProcess

# Basic images

X_train = np.array([0, 0.3, 2, 4])
y_train = np.sin(X_train)

X = np.arange(min(X_train) - 0.5, max(X_train) + 0.5, step=0.1)

GaussianProcess().fit(np.array([]),
        np.array([])).optimize_kernel().posterior(X).plot_posterior()

GaussianProcess(kernel=Matern(sigma=10, ro=5))\
        .posterior(X, X_train, y_train).plot_posterior(num_samples=0)\
        .optimize_kernel().posterior(X).plot_posterior()

for k in [SquaredExp(), RationalQuadratic()]: #, Linear()]:
    print(k)
    GaussianProcess(kernel=k).plot_prior(X)

for kernel in [RationalQuadratic(alpha=0.005)]: #, RationalQuadratic(), SquaredExp()]: #, Linear()]:
    GaussianProcess(kernel=kernel, noise=0.02)\
        .posterior(X, X_train, y_train).plot_posterior(num_samples=0)



################################################

X_train = np.array([0, 0.3, 2, 4])
y_train = np.sin(X_train)

X = np.arange(min(X_train) - 0.5, max(X_train) + 0.5, step=0.1)


# Kernel with rounding
gp = GaussianProcess()
gp.kernel = gp.kernel.with_round_indexes(np.array([0]))
gp.posterior(X, X_train, y_train).plot_posterior()

plt.title("Kernel with rounding")
plt.show()


# Different noise levels
for i, noise in enumerate([2, 0.2, 0.002, 0.0002]):
    gp = GaussianProcess().with_noise(noise).posterior(X, X_train, y_train)
    plt.subplot(2, 2, i + 1)

    gp.plot_posterior(num_samples=0, figure=False)

    plt.plot(X, np.sqrt(np.diag(gp.cov)), label="std")
    plt.plot(X, np.zeros_like(X), label="zero", c="k")

    if i == 0:
        plt.legend()

plt.show()
