import matplotlib.pyplot as plt
import numpy as np

from myopt.gaussian_process import GaussianProcess

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
