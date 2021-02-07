import math

import numpy as np
from delayed_bandit.simulation import Simulation


def bernoulli_etc_upper_bound(simulation: Simulation, n: int, m: int) -> float:
    """
    Calculate an upper bound for Explore-Then-Commit in the Bernoulli bandit setting.
    :param simulation: completed Simulation
    :param n: horizon
    :param m: number of exploration per arm
    :return: the upper bound
    """
    k = simulation.environment.num_arms()
    if m < 1 or m > math.floor(n / k):
        raise ValueError(f"n={n}, k={k}, m={m} don't satisfy 1 <= m <= n/k")

    regret = 0.0
    for arm in range(k):
        gap = simulation.environment.suboptimality_gap(arm)
        # ETC regret
        regret += m * gap + (n - m * k) * gap * math.exp(-m * math.pow(gap, 2))
        # delay-based penalty
        arm_rounds = np.where(simulation.arms[0:n] == arm)
        arm_delays = simulation.delays[0:n][arm_rounds]
        regret += gap * np.max(arm_delays)
    return regret


def bernoulli_etc_upper_bounds(
    simulation: Simulation, horizon: int, num_explorations: int, step: int
) -> np.ndarray:
    """
    Calculate upper bounds on every step-th round for the given horizon
    for Explore-Then-Commit in the Bernoulli bandit setting.
    """
    upper_bounds = np.zeros(horizon, dtype=np.float32)
    start_with = num_explorations * simulation.num_arms
    for n in range(start_with, horizon + 1, step):
        bound = bernoulli_etc_upper_bound(
            simulation=simulation, n=n, m=num_explorations
        )
        upper_bounds[n - 1] = bound
    upper_bounds[horizon - 1] = bernoulli_etc_upper_bound(
        simulation=simulation, n=horizon, m=num_explorations
    )
    return upper_bounds
