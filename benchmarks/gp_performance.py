import timeit
import numpy as np
from bopt.gaussian_process import GaussianProcess

noise = 0.2

# Noisy training data
X_train = 400*np.random.rand(100) - 200
y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

gp = GaussianProcess()

X = np.arange(-200, 200, 0.4)

t1 = timeit.Timer(lambda: gp.fit(X_train, y_train).optimize_kernel().posterior(X))
print(t1.timeit(number=10))


# X = np.arange(-200, 200, 0.2)
#
# t2 = timeit.Timer(lambda: gp.fit(X_train, y_train).optimize_kernel().posterior(X))
# print(t2.timeit(number=10))
