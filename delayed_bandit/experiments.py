import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from delayed_bandit.environments.bernoulli_bandit import BernoulliBandit
from delayed_bandit.bernoulli_etc_upper_bound import bernoulli_etc_upper_bounds
from delayed_bandit.policies.beta_thompson_sampling import BetaThompsonSampling
from delayed_bandit.policies.epsilon_greedy import EpsilonGreedy
from delayed_bandit.policies.etc import ETC
from delayed_bandit.policies.policy import Policy
from delayed_bandit.policies.uniform_random import UniformRandom
from delayed_bandit.simulation import simulate, Simulation


def bernoulli_experiments():
    """
    Run experiments for Bernoulli bandit with registered policies.

    Results from all settings (different delay samplings) are produced in CSV files for further analysis,
    one might want to run a significant number of experiments and aggregate it later by removing outliers
    and averaging results. Delay sampling is fixated among runs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", help="number of simulations", required=False, type=int, default=1
    )
    parser.add_argument(
        "--horizon",
        help="number of rounds to play",
        required=False,
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--etc-explorations",
        help="number of explorations in Explore-First",
        required=False,
        type=int,
        default=100,
    )
    parser.add_argument(
        "--egreedy-epsilon",
        help="epsilon in Epsilon-Greedy",
        required=False,
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--output", help="directory output path", required=False, type=str
    )
    parser.add_argument("--seed", help="random seed", required=False, type=int)
    args = parser.parse_args()

    def save(filename: str, array: np.ndarray):
        if not args.output:
            logging.warning(f"output path not provided, skip saving {filename}")
            return
        Path(args.output).mkdir(parents=True, exist_ok=True)
        np.savetxt(os.path.join(args.output, filename), array)

    seed = random.randrange(sys.maxsize)
    if args.seed:
        seed = args.seed
    logging.info(f"RNG seed of numpy: {seed}")
    rng = default_rng(seed)

    horizon = args.horizon
    means = [0.77, 0.8]  # the order is being shuffled at each run
    etc_num_explorations = args.etc_explorations
    egreedy_epsilon = args.egreedy_epsilon

    settings = [
        ("delay-const-0", np.zeros(horizon, dtype=np.int32)),
        ("delay-const-25", np.full(horizon, fill_value=25, dtype=np.int32)),
        ("delay-const-50", np.full(horizon, fill_value=50, dtype=np.int32)),
    ]

    for setting, delays in settings:
        for i in range(args.runs):
            rng.shuffle(means)
            environment = BernoulliBandit(means=means, rng=rng)
            num_arms = environment.num_arms()
            logging.info(f"Run #{i} with means: {means}")

            logging.info(f"ETC under {setting}")
            simulation = experiment(
                environment=environment,
                policy=ETC(num_arms=num_arms, num_explorations=etc_num_explorations),
                horizon=horizon,
                delays=delays,
            )
            upper_bounds = bernoulli_etc_upper_bounds(
                simulation=simulation,
                horizon=horizon,
                num_explorations=etc_num_explorations,
                step=100,
            )
            save(f"etcb-{setting}-regrets-{i}.csv", simulation.regrets())
            save(f"etcb-{setting}-upper-bounds-{i}.csv", upper_bounds)

            logging.info(f"Epsilon-Greedy under {setting}")
            simulation = experiment(
                environment=environment,
                policy=EpsilonGreedy(
                    num_arms=num_arms, epsilon=egreedy_epsilon, rng=rng
                ),
                horizon=horizon,
                delays=delays,
            )
            save(f"egreedyb-{setting}-regrets-{i}.csv", simulation.regrets())

            logging.info(f"Thompson Sampling under {setting}")
            simulation = experiment(
                environment=environment,
                policy=BetaThompsonSampling(num_arms=num_arms, rng=rng),
                horizon=horizon,
                delays=delays,
            )
            save(f"tsamplingb-{setting}-regrets-{i}.csv", simulation.regrets())

            logging.info(f"Uniform Random under {setting}")
            simulation = experiment(
                environment=environment,
                policy=UniformRandom(num_arms=num_arms, rng=rng),
                horizon=horizon,
                delays=delays,
            )
            save(f"urandomb-{setting}-regrets-{i}.csv", simulation.regrets())


def experiment(
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
    logging.info(f"> environment's optimal arm: {s.environment.optimal_arm()}")
    logging.info(f"> policy's empirically best arm: {s.policy.empirically_best_arm()}")
    logging.info(f"> regret: {s.regret(horizon)}")
    logging.info(f"> arm stats: {s.arms_stats()}")
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bernoulli_experiments()
