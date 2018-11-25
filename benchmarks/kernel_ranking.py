import matplotlib.pyplot as plt
from bayesian_optimization import bo_maximize_parallel, plot_2d_optim_result
from kernels import RationalQuadratic, Matern, SquaredExp
from myopt.opt_functions import get_opt_test_functions
from concurrent.futures import ThreadPoolExecutor


def main():
    executor = ThreadPoolExecutor()

    kernels = [SquaredExp()] #, Matern()]

    for n_iter in [25, 50, 100]:
        for n_parallel in [1,5,10]:
            for kernel in kernels:
                for optfun in get_opt_test_functions(executor):
                    res = bo_maximize_parallel(optfun.parallel_f, optfun.bounds, n_iter=n_iter,
                                               n_parallel=n_parallel, kernel=kernel)

                    plot_2d_optim_result(res)
                    plt.savefig(f"results/{optfun.name},kernel={kernel.name},niter={n_iter},npar={n_parallel},.png")


if __name__ == '__main__':
    main()