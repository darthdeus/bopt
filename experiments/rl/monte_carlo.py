#!/usr/bin/env python3
import numpy as np
import random

from collections import defaultdict

import cart_pole_evaluator


def main():
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=50, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    print(args)

    # Create the environment
    env = cart_pole_evaluator.environment()

    training = True

    Q = np.zeros([env.states, env.actions], dtype=np.float32)
    Q.fill(500)
    # Q.fill(1 / args.epsilon)

    C = np.zeros([env.states, env.actions], dtype=np.float32)

    eps_diff = (args.epsilon_final - args.epsilon) / float(args.episodes)
    eps_curr = args.epsilon

    while training:
        trajectory = []

        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            if random.random() < eps_curr:
                action = random.randint(0, env.actions - 1)
            else:
                action = np.argmax(Q[state]).item()

            next_state, reward, done, _ = env.step(action)

            trajectory.append([state, action, reward])

            state = next_state

        G = 0.0

        for state, action, reward in reversed(trajectory):
            G = args.gamma * G + reward
            # returns[(state, action)].append(G)
            # Q[state, action] = np.mean(returns[(state, action)]).item()

            C[state, action] += 1
            Q[state, action] += (G - Q[state, action])/C[state, action]

            state = next_state

        eps_curr += eps_diff

        if args.render_each and env.episode % args.render_each == 0:
            print(f"eps curr: {eps_curr}")

            # Evaluation episode
            state, done = env.reset(), False
            while not done:
                env.render()
                action = np.argmax(Q[state]).item()
                state, _, done, _ = env.step(action)

        if env.episode > args.episodes:
            break

    # Perform last 100 evaluation episodes
    for _ in range(100):
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state]).item()
            state, _, done, _ = env.step(action)

if __name__ == "__main__":
    from io import StringIO # Python3 use: from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    main()

    sys.stdout = old_stdout

    print(mystdout.getvalue().strip().split("\n")[-1].split(" ")[-1])

    # __import__('ipdb').set_trace()

    # examine mystdout.getvalue()
