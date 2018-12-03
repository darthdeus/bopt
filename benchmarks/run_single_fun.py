#!/usr/bin/env python
import argparse
from myopt.bayesian_optimization import bo_maximize
from myopt.kernels import get_kernel_by_name
from myopt.opt_functions import get_fun_by_name
from myopt.acquisition_functions import get_acquisition_fn_by_name

parser = argparse.ArgumentParser()
parser.add_argument("--fun", type=str, help="Function to optimize")
parser.add_argument("--n_iter", type=int, help="Number of iterations")
parser.add_argument("--kernel", type=str, help="Kernel")
parser.add_argument("--acquisition-fn", type=str, default="ei", help="Acquisition function")
args = parser.parse_args()


print(f"Running: {args}")

fun = get_fun_by_name(args.fun)
kernel = get_kernel_by_name(args.kernel)
acquisition_function = get_acquisition_fn_by_name(args.acquisition_fn)

result = bo_maximize(fun, fun.bounds, kernel, use_tqdm=False, n_iter=args.n_iter,
                     acquisition_function=acquisition_function)

print(result)

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
