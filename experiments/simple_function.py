#!/usr/bin/env python3

if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="relu", type=str, help="activation")
    parser.add_argument("--x", default=1.0, type=float, help="X")
    parser.add_argument("--y", default=1.0, type=float, help="Y")
    parser.add_argument("--z", default=1.0, type=float, help="Z")
    parser.add_argument("--w", default="sigmoid", type=str, help="W")
    args = parser.parse_args()

    import math
    import random
    import time

    if args.w == "sigmoid":
        args.w = 1.0
    elif args.w == "relu":
        args.w = 0.0
    elif args.w == "tanh":
        args.w = -1.0
    else:
        args.w = 2.0

    result = math.sin(args.x) * args.y**0.3 \
            + random.random() * 0.05 + 2*math.cos(args.z) * math.cos(args.w)

    # result = args.x + 2 * args.y - args.z**2
    print("RESULT={}".format(result))
