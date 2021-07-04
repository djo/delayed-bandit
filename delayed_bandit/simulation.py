import queue
from typing import List, Tuple

import numpy as np

from delayed_bandit.environments.environment import Environment
from delayed_bandit.policies.policy import Policy


class Simulation:
    def __init__(self, horizon: int, environment: Environment, policy: Policy, delays: np.ndarray):
        self.horizon = horizon
        self.num_arms = environment.num_arms()
        self.environment = environment
        self.policy = policy
        # arms pulled in the environment on these rounds (indices in the array)
        self.arms = np.full(horizon, fill_value=-1, dtype=np.int32)
        # rewards that supposed to be returned on these rounds in case of no delay,
        # taken into account for the regret calculation if the delay doesn't extend the horizon
        self.rewards = np.zeros(horizon, dtype=np.float32)
        # delays on each round
        self.delays = delays
        # FIFO queue per arm with realized rewards
        self.queues: List[queue.Queue] = [queue.Queue() for _ in range(self.num_arms)]
        # rewards to be realized at that round in tuple (arm, reward)
        self.future_rewards: List[List[Tuple[int, float]]] = [[] for _ in range(horizon)]

    def regret(self, n: int) -> float:
        """
        Return a cumulative regret after n rounds.
        """
        _, optimal_mean = self.environment.optimal_arm()
        return n * optimal_mean - sum(self.rewards[0:n])

    def regrets(self) -> np.ndarray:
        res = np.zeros(self.horizon, dtype=np.float32)
        _, optimal_mean = self.environment.optimal_arm()
        prev_regret = 0.0
        for i in range(self.horizon):
            res[i] = prev_regret + optimal_mean - self.rewards[i]
            prev_regret = res[i]
        return res

    def arms_stats(self) -> np.ndarray:
        stats = np.zeros(self.num_arms, dtype=np.int32)
        arms, counts = np.unique(self.arms, return_counts=True)
        for a in arms:
            stats[a] = counts[a]
        return stats


def simulate(horizon: int, environment: Environment, policy: Policy, delays: np.ndarray):
    simulation = Simulation(horizon=horizon, environment=environment, policy=policy, delays=delays)
    # simulated round for the policy to emulate sequential decision making
    policy_t = 0
    arm = policy.choice(policy_t)

    for t in range(horizon):
        while not simulation.queues[arm].empty():
            reward = simulation.queues[arm].get()
            policy.feed_reward(t=policy_t, arm=arm, reward=reward)
            policy_t += 1
            arm = policy.choice(policy_t)

        reward = environment.pull(arm)
        simulation.rewards[t] = reward
        simulation.arms[t] = arm

        if t + delays[t] < horizon:
            simulation.future_rewards[t + delays[t]].append((arm, reward))

        # add rewards delayed to be realized in the current round if any
        for a, r in simulation.future_rewards[t]:
            simulation.queues[a].put(r)

        # trigger reward feeding again for the case of no delay in this round
        while not simulation.queues[arm].empty():
            reward = simulation.queues[arm].get()
            policy.feed_reward(t=policy_t, arm=arm, reward=reward)
            policy_t += 1
            arm = policy.choice(policy_t)

    return simulation
