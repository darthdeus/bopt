import numpy as np
import random
import concurrent.futures
from myopt.bayesian_optimization import bo_maximize, Float, Integer


executor = concurrent.futures.ThreadPoolExecutor(1)

bounds = [Float(-1.0, 2.0)]
noise = 0.2

def f(X):
    return -np.sin(3*X) - X**2 + 0.7*X

def parallel_f(x):
    return executor.submit(noisy_f, x)

X_init = np.array([[-0.9], [1.1]])
y_init = f(X_init)

# Dense grid of points within bounds
X_true = np.arange(bounds[0].low, bounds[0].high, 0.01).reshape(-1, 1)

# Noise-free objective function values at X 
y_true = f(X_true)

noisy_f = lambda x: f(x).item() + random.random()*0.01

bo_maximize(noisy_f, bounds, n_iter=50)
