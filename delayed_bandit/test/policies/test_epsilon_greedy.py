import numpy as np
import pytest
from numpy.random import default_rng, PCG64, Generator

from delayed_bandit.policies.epsilon_greedy import EpsilonGreedy


class TestEpsilonGreedy:
    def test_choice_arm_to_explore(self):
        greedy = EpsilonGreedy(num_arms=2, epsilon=1.0, rng=Generator(PCG64(40)))
        greedy.cumulative_rewards = np.array([1.0, 1.0], dtype=np.float32)
        greedy.arms_stats = np.array([2, 1])
        assert greedy.choice(0) == 0

    def test_choice_arm_to_exploit(self):
        greedy = EpsilonGreedy(num_arms=2, epsilon=0.0, rng=default_rng())
        greedy.cumulative_rewards = np.array([1.0, 1.0], dtype=np.float32)
        greedy.arms_stats = np.array([2, 1])
        assert greedy.choice(0) == 1

    def test_feed_reward(self):
        greedy = EpsilonGreedy(num_arms=2, epsilon=0.0, rng=Generator(PCG64(42)))
        assert greedy.choice(0) == 1
        with pytest.raises(ValueError, match=r"Expected the reward for arm 1, but got for 0"):
            greedy.feed_reward(t=0, arm=0, reward=0.0)
        greedy.feed_reward(t=0, arm=1, reward=0.0)

    def test_empirically_best_arm_uniformly_when_no_rewards_yet(self):
        greedy = EpsilonGreedy(num_arms=3, epsilon=0.1, rng=Generator(PCG64(42)))
        assert greedy.empirically_best_arm() == (0, 0.0)

    def test_empirically_best_arm(self):
        greedy = EpsilonGreedy(num_arms=3, epsilon=0.1, rng=default_rng())
        greedy.cumulative_rewards = np.array([0.0, 10.0, 1.0], dtype=np.float32)

        greedy.arms_stats = np.array([0, 11, 1])
        assert greedy.empirically_best_arm() == (2, 1.0)

        greedy.arms_stats = np.array([0, 10, 1])
        assert greedy.empirically_best_arm() == (1, 1.0)

        greedy.arms_stats = np.array([0, 2, 1])
        assert greedy.empirically_best_arm() == (1, 5.0)
