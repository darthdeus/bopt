#!/usr/bin/env python3

if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--x", default=1.0, type=float, help="X")
    parser.add_argument("--y", default=1.0, type=float, help="Y")
    parser.add_argument("--z", default=1.0, type=float, help="Z")
    args = parser.parse_args()

    import math
    import random

    result = math.sin(args.x) * math.sin(args.y) + random.random() * 0.05

    result = args.x + 2 * args.y - args.z**2
    print(result)
