import numpy as np
import pytest
from numpy.random import default_rng, PCG64, Generator

from delayed_bandit.policies.uniform_random import UniformRandom


class TestUniformRandom:
    def test_choice(self):
        greedy = UniformRandom(num_arms=2, rng=Generator(PCG64(40)))
        assert greedy.choice(0) == 1

    def test_feed_reward(self):
        uniform = UniformRandom(num_arms=2, rng=Generator(PCG64(40)))
        assert uniform.choice(0) == 1
        with pytest.raises(
            ValueError, match=r"Expected the reward for arm 1, but got for 0"
        ):
            uniform.feed_reward(t=0, arm=0, reward=0.0)
        uniform.feed_reward(t=0, arm=1, reward=0.0)

    def test_empirically_best_arm(self):
        uniform = UniformRandom(num_arms=3, rng=default_rng())
        uniform.cumulative_rewards = np.array([0.0, 10.0, 1.0], dtype=np.float32)

        uniform.arms_stats = np.array([0, 11, 1])
        assert uniform.empirically_best_arm() == (2, 1.0)

        uniform.arms_stats = np.array([0, 10, 1])
        assert uniform.empirically_best_arm() == (1, 1.0)

        uniform.arms_stats = np.array([0, 2, 1])
        assert uniform.empirically_best_arm() == (1, 5.0)
