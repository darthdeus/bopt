def bo_maximize(f: Callable[[np.array], float], params: List[Hyperparameter],
                kernel: Kernel = SquaredExp(), acquisition_function=expected_improvement,
                x_0: np.ndarray = None, gp_noise: float = 0,
                n_iter: int = 8, callback: Callable = None,
                optimize_kernel=True, use_tqdm=True) -> OptimizationResult:

    bounds = [p.range for p in params]

    if x_0 is None:
        x_0 = default_from_bounds(bounds)
    else:
        for i, bound in enumerate(bounds):
            assert bound.low <= x_0[i] <= bound.high, f"x_0 not in bounds, {bound} at {i}"

    kernel = kernel.with_bounds(bounds)

    y_0 = f(x_0)

    # TODO: handle numpy rank-0 tensors
    assert type(y_0) == float, f"f(x) must return a float, got type {type(y_0)}, value: {y_0}"

    X_sample = np.array([x_0])
    y_sample = np.array([y_0])

    iter_target = range(n_iter - 1)
    if use_tqdm:
        from tqdm import tqdm
        iter_target = tqdm(iter_target)

    for iter in iter_target:
        gp = GaussianProcess(kernel=kernel, noise=gp_noise).fit(X_sample, y_sample)
        if optimize_kernel:
            gp = gp.optimize_kernel()

        x_next = propose_location(acquisition_function, gp, y_sample.max(), bounds)

        y_next = f(x_next)

        if callback is not None:
            callback(iter, acquisition_function, gp, X_sample, y_sample, x_next, y_next)

        X_sample = np.vstack((X_sample, x_next))
        y_sample = np.hstack((y_sample, y_next))

    max_y_ind = y_sample.argmax()
    # print("max_x", X_sample[max_y_ind], "max max", y_sample.max())

    return OptimizationResult(X_sample,
                              y_sample,
                              best_x=X_sample[y_sample.argmax()],
                              best_y=y_sample.max(),
                              params=params,
                              kernel=kernel.copy(),
                              n_iter=n_iter,
                              opt_fun=f)

