#!/usr/bin/env python3

if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="relu", type=str, help="activation")
    parser.add_argument("--x", default=1.0, type=float, help="X")
    parser.add_argument("--y", default=1.0, type=float, help="Y")
    parser.add_argument("--z", default=1.0, type=float, help="Z")
    parser.add_argument("--w", default=1.0, type=float, help="W")
    args = parser.parse_args()

    import math
    import random
    import time

    if args.activation == "sigmoid":
        act = 1.0
    elif args.activation == "relu":
        act = 2.0
    elif args.activation == "tanh":
        act = -1.0
    else:
        raise NotImplementedError()

    result = act * (args.x + math.log2(args.y) + args.w ** 0.3 \
            + random.random() * 0.05 + 2*args.z)

    # result = args.x + 2 * args.y - args.z**2
    print("RESULT={}".format(result))
