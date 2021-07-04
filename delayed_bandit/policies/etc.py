from typing import Tuple

import numpy as np

from delayed_bandit.policies.policy import Policy


class ETC(Policy):
    def __init__(self, num_arms: int, num_explorations: int):
        """
        Create Explore-First policy for stochastic bandit environment.
        :param num_arms: number of arms of the bandit
        :param num_explorations: number of times the algorithm explores each arm
        """
        self._num_arms = num_arms
        self._num_explorations = num_explorations
        self._current_arm = -1
        self.cumulative_rewards = np.zeros(num_arms, dtype=np.float32)

    def choice(self, t: int) -> int:
        if t < self._num_explorations * self._num_arms:
            self._current_arm = t % self._num_arms
            return self._current_arm
        self._current_arm, _ = self.empirically_best_arm()
        return self._current_arm

    def feed_reward(self, t: int, arm: int, reward: float):
        if t >= self._num_explorations * self._num_arms:
            # ignore rewards during exploitation as the optimal arm is already set
            self._current_arm = -1
            return
        if arm != self._current_arm:
            raise ValueError(f"Expected the reward for arm {self._current_arm}, but got for {arm}")
        self.cumulative_rewards[arm] += reward
        return

    def empirically_best_arm(self) -> Tuple[int, float]:
        arm = np.argmax(self.cumulative_rewards)
        return int(arm), self.cumulative_rewards[arm] / self._num_explorations
