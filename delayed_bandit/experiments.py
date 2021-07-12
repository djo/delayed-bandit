import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
from numpy.random import default_rng

from delayed_bandit.environments.bernoulli_bandit import BernoulliBandit
from delayed_bandit.bernoulli_etc_upper_bound import bernoulli_etc_upper_bounds
from delayed_bandit.policies.beta_thompson_sampling import BetaThompsonSampling
from delayed_bandit.policies.epsilon_greedy import EpsilonGreedy
from delayed_bandit.policies.etc import ETC
from delayed_bandit.policies.policy import Policy
from delayed_bandit.policies.ucb import UCB
from delayed_bandit.policies.uniform_random import UniformRandom
from delayed_bandit.simulation import simulate, Simulation


def bernoulli_experiments():
    """
    Run experiments for Bernoulli bandit with registered policies.

    Results from all settings (different delay samplings) are produced in CSV files
    for further analysis, one might want to run a significant number of experiments
    and aggregate it later by removing outliers and averaging results.
    Delay sampling is fixated among runs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", help="number of simulations", required=False, type=int, default=1000
    )
    parser.add_argument(
        "--horizon",
        help="number of rounds to play",
        required=False,
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--etc-explorations",
        help="number of explorations in Explore-First",
        required=False,
        type=int,
        default=620,
    )
    parser.add_argument(
        "--egreedy-epsilon",
        help="epsilon in Epsilon-Greedy",
        required=False,
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--ucb-alpha",
        help="alpha in Upper Confidence Bound",
        required=False,
        type=float,
        default=0.1,
    )
    parser.add_argument("--output", help="directory output path", required=False, type=str)
    parser.add_argument(
        "--seed",
        help="random seed",
        required=False,
        type=int,
        default=random.randrange(sys.maxsize),
    )
    args = parser.parse_args()

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    def save(filename: str, array: np.ndarray):
        if not args.output:
            logging.warning(f"output path not provided, skip saving {filename}")
            return
        np.savetxt(os.path.join(args.output, filename), array)

    seed = args.seed
    logging.info(f"RNG seed of numpy: {seed}")
    rng = default_rng(seed)

    horizon = args.horizon
    means = [0.77, 0.8]  # the order is being shuffled at each run
    num_arms = len(means)
    etc_num_explorations = args.etc_explorations
    egreedy_epsilon = args.egreedy_epsilon
    ucb_alpha = args.ucb_alpha

    settings = [
        ("delay-const-0", np.zeros(horizon, dtype=np.int32)),
        # ("delay-const-25", np.full(horizon, fill_value=25, dtype=np.int32)),
        # ("delay-const-50", np.full(horizon, fill_value=50, dtype=np.int32)),
    ]

    def run(policy_builder: Callable[[], Policy]):
        for setting, delays in settings:
            simulations = []
            for i in range(args.runs):
                rng.shuffle(means)
                environment = BernoulliBandit(means=means, rng=rng)
                policy = policy_builder()
                logging.info(f"= {policy.name()} under {setting} #{i}")
                simulation = _experiment(
                    environment=environment,
                    policy=policy,
                    horizon=horizon,
                    delays=delays,
                )
                simulations.append(simulation)
            regrets = _aggregate(list(map(lambda s: s.regrets(), simulations)))
            save(f"{policy_builder().name()}-{setting}-regrets.csv", regrets)

    # run policies in all settings aggregating results over runs
    run(lambda: UniformRandom(num_arms=num_arms, rng=rng))
    run(lambda: ETC(num_arms=num_arms, num_explorations=1))
    run(lambda: ETC(num_arms=num_arms, num_explorations=etc_num_explorations))
    run(lambda: EpsilonGreedy(num_arms=num_arms, epsilon=egreedy_epsilon, rng=rng))
    run(lambda: UCB(num_arms=num_arms, alpha=ucb_alpha, rng=rng))
    run(lambda: BetaThompsonSampling(num_arms=num_arms, rng=rng))

    # run upper bound calculations in all settings
    for setting, delays in settings:
        rng.shuffle(means)
        environment = BernoulliBandit(means=means, rng=rng)
        policy = ETC(num_arms=num_arms, num_explorations=etc_num_explorations)
        logging.info(f"= {policy.name()} under {setting} upper bound calculation")
        simulation = _experiment(
            environment=environment,
            policy=policy,
            horizon=horizon,
            delays=delays,
        )
        upper_bounds = bernoulli_etc_upper_bounds(
            simulation=simulation,
            horizon=horizon,
            num_explorations=etc_num_explorations,
            step=100,
        )
        save(f"{policy.name()}-{setting}-upper-bounds.csv", upper_bounds)


def _aggregate(results: List[np.ndarray]):
    arrays = np.array(results)
    final_regrets = np.array([a[-1] for a in arrays])
    final_regrets_percentiles = np.percentile(final_regrets, [0, 10, 30, 50, 70, 90, 100])
    logging.info(f'= final regrets percentiles: {final_regrets_percentiles}')
    lower, upper = np.percentile(final_regrets, [10, 90])
    if lower < upper:
        filtered = arrays[np.where((final_regrets > lower) & (final_regrets < upper))]
        if len(filtered) > 0:
            logging.info(
                f'= filtered by final regret lower {lower}, upper {upper} new size: {len(filtered)}'
            )
            arrays = filtered
    return np.mean(arrays, axis=0)


def _experiment(
    environment: BernoulliBandit,
    policy: Policy,
    horizon: int,
    delays: np.ndarray,
) -> Simulation:
    s = simulate(
        horizon=horizon,
        environment=environment,
        policy=policy,
        delays=delays,
    )
    logging.info(f"> environment's means: {s.environment.means()}")
    logging.info(f"> environment's optimal arm: {s.environment.optimal_arm()}")
    logging.info(f"> policy's empirically best arm: {s.policy.empirically_best_arm()}")
    logging.info(f"> regret: {s.regret(horizon)}")
    logging.info(f"> arm stats: {s.arms_stats()}")
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bernoulli_experiments()
