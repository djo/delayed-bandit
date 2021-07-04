import numpy as np
from numpy.random import default_rng
from delayed_bandit.policies.ucb import UCB
import pytest


class TestUCB:
    def test_choice_with_large_mean(self):
        ucb = UCB(num_arms=2, alpha=0.5, rng=default_rng())
        ucb.cumulative_rewards = np.array([0.0, 10.0], dtype=np.float32)
        ucb.arms_stats = np.array([1, 1])
        assert ucb.choice(2) == 1

    def test_choice_with_large_confidence_radius(self):
        ucb = UCB(num_arms=2, alpha=0.5, rng=default_rng())
        ucb.cumulative_rewards = np.array([0.1, 0.1], dtype=np.float32)
        ucb.arms_stats = np.array([1, 10])
        assert ucb.choice(2) == 0

    def test_feed_reward(self):
        ucb = UCB(num_arms=2, alpha=0.5, rng=default_rng())
        assert ucb.choice(0) == 0
        with pytest.raises(ValueError, match=r"Expected the reward for arm 0, but got for 1"):
            ucb.feed_reward(t=0, arm=1, reward=0.0)
        ucb.feed_reward(t=0, arm=0, reward=0.0)

    def test_empirically_best_arm(self):
        ucb = UCB(num_arms=3, alpha=0.5, rng=default_rng())
        ucb.cumulative_rewards = np.array([0.0, 10.0, 1.0], dtype=np.float32)

        ucb.arms_stats = np.array([0, 11, 1])
        assert ucb.empirically_best_arm() == (2, 1.0)

        ucb.arms_stats = np.array([0, 10, 1])
        assert ucb.empirically_best_arm() == (1, 1.0)

        ucb.arms_stats = np.array([0, 2, 1])
        assert ucb.empirically_best_arm() == (1, 5.0)
