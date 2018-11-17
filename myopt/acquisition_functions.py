import numpy as np
from scipy.stats import norm

from .gaussian_process import GaussianProcess


def expected_improvement(gp: GaussianProcess, X: np.ndarray, f_s: float, xi: float=0.01) -> np.ndarray:
    mu, sigma = gp.posterior(X).mu_std()

    sigma = sigma.reshape(-1, 1)

    improvement = mu - f_s - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    return ei


# def expected_improvement(gp, X, X_sample, xi=0.01) -> np.ndarray:
#     # gp = GaussianProcess(kernel=kernel).fit(X_sample, y_sample)
#     # if optimize_kernel:
#     #     gp = gp.optimize_kernel()
#
#     mu, sigma = gp.posterior(X).mu_std()
#     # mu, sigma = gp_reg(X_sample, y_sample, X, return_std=True)
#
#     mu_sample, _ = gp.posterior(X_sample).mu_std()
#     mu_sample_opt = np.max(mu_sample)
#
#     sigma = sigma#.reshape(-1, X.shape[1])
#
#     # Needed for noise-based model,
#     # otherwise use np.max(y_sample).
#     # See also section 2.4 in [...]
#
#     with np.errstate(divide='warn'):
#         imp = mu - mu_sample_opt - xi
#         Z = imp / sigma
#         ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
#         ei[sigma == 0.0] = 0.0
#
#     return ei
