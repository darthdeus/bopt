import numpy as np
from scipy.stats import norm


# X_sample, y_sample, kernel,
def expected_improvement(X, gp, X_sample, xi=0.01) -> np.ndarray:
    # gp = GaussianProcess(kernel=kernel).fit(X_sample, y_sample)
    # if optimize_kernel:
    #     gp = gp.optimize_kernel()

    mu, sigma = gp.posterior(X).mu_std()
    # mu, sigma = gp_reg(X_sample, y_sample, X, return_std=True)

    mu_sample, sigma_sample = gp.posterior(X_sample).mu_std()
    # mu_sample, sigma_sample = gp_reg(X_sample, y_sample, X_sample, return_std=True)

    sigma = sigma.reshape(-1, X_sample.shape[1])

    # Needed for noise-based model,
    # otherwise use np.max(y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei
