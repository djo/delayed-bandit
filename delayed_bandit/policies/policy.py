from typing import Protocol, Tuple


class Policy(Protocol):
    def choice(self, t: int) -> int:
        """
        Choose next arm to play on the given round.

        :param t: current round [0, n), where n is the horizon
        :return: an arm index to play [0, k), where k is the number of arms
        """
        ...

    def feed_reward(self, t: int, arm: int, reward: float):
        """
        Feed the realised reward to the policy.

        :param t: a given round [0, n)
        :param arm: an arm index to play [0, k)
        :param reward: realised reward
        :return: nothing
        """
        ...

    def empirically_best_arm(self) -> Tuple[int, float]:
        """
        Return an arm and its mean with the largest sample mean among others,
        breaking ties by choosing the smallest index.
        """
        ...
