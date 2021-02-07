from typing import List, Tuple

import numpy as np
from numpy.testing import assert_array_equal

from delayed_bandit.simulation import simulate, Simulation


class TestSimulation:
    def test_regret(self):
        environment = EnvironmentStub(optimal_arm=(0, 0.5))
        delays = np.zeros(5, dtype=np.int32)
        s = Simulation(horizon=5, environment=environment, policy=None, delays=delays)
        s.rewards = np.array([0, 0, 1, 1, 0], dtype=np.float32)

        assert s.regret(1) == 0.5
        assert s.regret(2) == 1.0
        assert s.regret(3) == 0.5
        assert s.regret(4) == 0.0
        assert s.regret(5) == 0.5

        assert_array_equal(s.regrets(), [0.5, 1.0, 0.5, 0.0, 0.5])


def test_simulate_with_no_delays():
    s = simulate(
        horizon=5,
        environment=BernoulliBanditStub(rewards=[1, 0], num_arms=2),
        policy=ETCStub(2),
        delays=np.zeros(5, dtype=np.int32),
    )

    assert_array_equal(s.arms, [0, 1, 0, 1, 0])
    assert_array_equal(s.rewards, [1.0, 0.0, 1.0, 0.0, 1.0])
    assert_array_equal(s.arms_stats(), [3, 2])


def test_simulate_with_delays():
    s = simulate(
        horizon=6,
        environment=BernoulliBanditStub(rewards=[1, 0], num_arms=2),
        policy=ETCStub(2),
        delays=np.full(6, fill_value=1, dtype=np.int32),
    )

    assert_array_equal(s.arms, [0, 0, 1, 1, 1, 0])
    assert_array_equal(s.rewards, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert_array_equal(s.arms_stats(), [3, 3])


class ETCStub:
    def __init__(self, k: int):
        self.k = k

    def choice(self, t: int) -> int:
        return t % self.k  # simply choose all arms in turn

    def feed_reward(self, t: int, arm: int, reward: float):
        return


class BernoulliBanditStub:
    def __init__(self, rewards: List[int], num_arms=2):
        self._rewards = rewards  # fixated rewards per arm
        self._num_arms = num_arms

    def pull(self, arm) -> int:
        return self._rewards[arm]

    def num_arms(self) -> int:
        return self._num_arms


class EnvironmentStub:
    def __init__(self, optimal_arm: Tuple[int, float]):
        self._optimal_arm = optimal_arm

    def optimal_arm(self):
        return self._optimal_arm

    def num_arms(self):
        return 1
