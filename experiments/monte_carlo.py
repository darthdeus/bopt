#!/usr/bin/env python3

OPTIMIZE_HYPERPARAMS = False

import random
# from argparse import Namespace
# from typing import NamedTuple, List
# if OPTIMIZE_HYPERPARAMS:
#     from skopt import gp_minimize

import numpy as np

import cart_pole_evaluator
# from gym_evaluator import GymEnvironment


# class SAR:
#     state: int
#     action: int
#     reward: float


def update_timestep(sar, G: float, Q: np.ndarray, C: np.ndarray) -> float:
    s, a, r = sar

    G = args.gamma * G + r
    C[s, a] += 1

    Qsa = Q[s, a]
    Csa = C[s, a]

    Q[s, a] = Qsa + (1.0 / Csa * (G - Qsa))

    return G


def train(args, env, Q, C) -> None:
    for episode in range(args.episodes):
        state, done = env.reset(False), False
        sar_tups = []

        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            if np.random.rand() < args.epsilon:
                action = random.choice(range(env.actions))
            else:
                action = Q.argmax(axis=1)[state]

            next_state, reward, done, _ = env.step(action)
            sar_tups.append((state, action, reward))
            state = next_state

        G = 0

        for t in reversed(range(len(sar_tups))):
            G = update_timestep(sar_tups[t], G, Q, C)


def evaluate(args, env, Q) -> float:
    for _ in range(100):
        # Perform a training episode
        state, done = env.reset(True), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            action = Q.argmax(axis=1)[state]
            try:
                next_state, reward, done, info = env.step(action)
            # FUJ ale lip to neumim :P
            except UnicodeError as e:
                # print("caught ", e.args[0])
                return e.args[0]

            state = next_state


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.45, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.45, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment

    # TODO: Implement Monte-Carlo RL algorithm.
    #
    # The overall structure of the code follows.


    def target_fn(x) -> float:
        env = cart_pole_evaluator.environment()

        epsilon, epsilon_final, gamma = x
        args.epsilon = epsilon
        args.epsilon_final = epsilon_final
        args.gamma = gamma

        Q = np.zeros((env.states, env.actions), dtype=np.float32)
        C = np.zeros_like(Q)

        train(args, env, Q, C)

        # Perform last 100 evaluation episodes
        mean_value = evaluate(args, env, Q)

        return -mean_value


    if OPTIMIZE_HYPERPARAMS:
        pass
        # best = gp_minimize(target_fn, [
        #     (0.001, 1),
        #     (0.001, 1),
        #     (0.001, 1)
        # ], n_calls=20)
        #
        # print(best)
    else:
        x = [0.471958418783538, 0.06929413737303698, 0.5808494337551399]
        target_fn(x)
