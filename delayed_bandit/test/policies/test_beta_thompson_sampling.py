import numpy as np
import pytest
from numpy.random import default_rng, Generator, PCG64

from delayed_bandit.policies.beta_thompson_sampling import BetaThompsonSampling


class TestBetaThompsonSampling:
    def test_choice_arm_to_explore(self):
        greedy = BetaThompsonSampling(num_arms=2, rng=Generator(PCG64(40)))
        assert greedy.choice(0) == 0

    def test_feed_reward(self):
        greedy = BetaThompsonSampling(num_arms=2, rng=Generator(PCG64(42)))
        assert greedy.choice(0) == 1
        with pytest.raises(ValueError, match=r"Expected the reward for arm 1, but got for 0"):
            greedy.feed_reward(t=0, arm=0, reward=0.0)
        greedy.feed_reward(t=0, arm=1, reward=0.0)

    def test_empirically_best_arm(self):
        ts = BetaThompsonSampling(num_arms=3, rng=default_rng())
        ts.cumulative_rewards = np.array([0, 10, 1], dtype=np.int32)

        ts.arms_stats = np.array([0, 11, 1])
        assert ts.empirically_best_arm() == (2, 1.0)

        ts.arms_stats = np.array([0, 10, 1])
        assert ts.empirically_best_arm() == (1, 1.0)

        ts.arms_stats = np.array([0, 4, 1])
        assert ts.empirically_best_arm() == (1, 2.5)
