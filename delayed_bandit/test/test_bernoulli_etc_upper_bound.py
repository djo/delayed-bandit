import numpy as np
import pytest

from delayed_bandit.bernoulli_etc_upper_bound import bernoulli_etc_upper_bound


def test_bernoulli_etc_upper_bound_with_one_arm():
    s = SimulationStub(
        num_arms=1,
        arms=np.array([0, 0, 0], dtype=np.int32),
        gaps=[0.0],
        delays=np.array([0, 0, 0], dtype=np.int32),
    )
    assert bernoulli_etc_upper_bound(simulation=s, n=3, m=1) == 0.0


def test_bernoulli_etc_upper_bound_with_two_arms():
    s = SimulationStub(
        num_arms=2,
        arms=np.array([0, 1, 0], dtype=np.int32),
        gaps=[0.0, 0.9],
        delays=np.array([0, 0, 0], dtype=np.int32),
    )
    pytest.approx(bernoulli_etc_upper_bound(simulation=s, n=3, m=1)) == 1.3

    s = SimulationStub(
        num_arms=2,
        arms=np.array([0, 1, 0], dtype=np.int32),
        gaps=[0.0, 0.9],
        delays=np.array([0, 10, 0], dtype=np.int32),
    )
    pytest.approx(bernoulli_etc_upper_bound(simulation=s, n=3, m=1)) == 10.3


class SimulationStub:
    def __init__(self, num_arms, arms, gaps, delays):
        self.environment = self
        self.arms = arms
        self.delays = delays
        self._num_arms = num_arms
        self._gaps = gaps

    def num_arms(self):
        return self._num_arms

    def suboptimality_gap(self, arm):
        return self._gaps[arm]
