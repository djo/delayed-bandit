from typing import List, Tuple, Optional

import numpy as np
from numpy.random import Generator

from delayed_bandit.environments.environment import Environment


class BernoulliBandit(Environment):
    def __init__(self, means: List[float], rng: Generator):
        self._rng = rng
        self._k = len(means)
        self._means = np.array(means)
        if np.any((self._means < 0) | (self._means > 1)):
            raise ValueError(f"Means {means} must be in [0, 1]")
        self._optimal_arm: Optional[Tuple[int, float]] = None

    def pull(self, arm: int) -> float:
        p = self._means[arm]
        return self._rng.binomial(n=1, p=p)

    def optimal_arm(self) -> Tuple[int, float]:
        if self._optimal_arm:
            return self._optimal_arm
        arm = np.argmax(self._means)
        self._optimal_arm = (int(arm), self._means[arm])
        return self._optimal_arm

    def suboptimality_gap(self, arm: int) -> float:
        _, optimal_mean = self.optimal_arm()
        mean = self._means[arm]
        return abs(optimal_mean - mean)

    def num_arms(self) -> int:
        return self._k

    def means(self) -> np.ndarray:
        return self._means
