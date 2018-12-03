import numpy as np
from myopt.bayesian_optimization import bo_maximize
from myopt.opt_functions import get_opt_test_functions
from myopt.kernels import SquaredExp, Matern

from joblib import Parallel, delayed


opt_functions = get_opt_test_functions()
kernels = [SquaredExp(), Matern()]


for ei in ["ei", "pi"]:
    for opt_fun in opt_functions:
        for n_iter in [10, 25, 50, 100]:
            for kernel in kernels:
                print(f"--acquisition-fn={ei} --fun={opt_fun.name} --n_iter={n_iter} --kernel={kernel.name}")

#
# results = Parallel(n_jobs=-1)(
#         delayed(bo_maximize)(opt_fun, opt_fun.bounds, kernel, use_tqdm=False, n_iter=n_iter)
#         for (opt_fun, n_iter, kernel) in combinations
#         )
#
# strs = [f"{res.opt_fun.name}\t{res.opt_fun.max}\t{res.n_iter}\t{res.kernel.name}\t{round(res.best_y, 3)}"
#         for res in results]
#
# print("\n".join(strs))




# for opt_fun in opt_functions:
#     print(f"FUN: {opt_fun.name}\topt: {opt_fun.max}")
#     for n_iter in [10, 25, 50]:
#         print(f"n_iter={n_iter}")
#         for kernel in kernels:
#             result = bo_maximize(opt_fun.f, opt_fun.bounds,
#                     kernel, use_tqdm=False, n_iter=n_iter)
#
#             print(f"{kernel.name}\t{round(result.best_y, 3)}")
#
#         print("")
#
#     print("*************************************")
