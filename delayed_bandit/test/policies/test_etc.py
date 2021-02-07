import numpy as np

import pytest

from delayed_bandit.policies.etc import ETC


class TestETC:
    def test_choice(self):
        etc = ETC(num_arms=2, num_explorations=1)

        # exploration phase
        assert etc.choice(t=0) == 0
        etc.feed_reward(t=0, arm=0, reward=1.0)
        assert etc.choice(t=1) == 1
        etc.feed_reward(t=1, arm=1, reward=1.1)

        # exploitation phase
        assert etc.choice(t=2) == 1
        etc.feed_reward(t=3, arm=1, reward=0.0)
        assert etc.choice(t=3) == 1

    def test_feed_reward(self):
        etc = ETC(num_arms=1, num_explorations=1)
        assert etc.choice(0) == 0
        with pytest.raises(
            ValueError, match=r"Expected the reward for arm 0, but got for 1"
        ):
            etc.feed_reward(t=0, arm=1, reward=0.0)
        etc.feed_reward(t=0, arm=0, reward=0.0)

    def test_empirically_best_arm(self):
        etc = ETC(num_arms=3, num_explorations=4)
        etc.cumulative_rewards = np.array([1.0, 2.0, 2.0])
        assert etc.empirically_best_arm() == (1, 0.5)
