from typing import Protocol, Tuple


class Environment(Protocol):
    def pull(self, arm: int) -> float:
        """
        Draw the reward from a given arm.

        :param arm: an arm index
        :return: a realised reward
        """
        ...

    def optimal_arm(self) -> Tuple[int, float]:
        """
        Return an arm with the largest population mean,
        breaking ties by choosing the smallest index.

        :return: a tuple of arm index and its expected mean
        """
        ...

    def suboptimality_gap(self, arm: int) -> float:
        """
        Return the difference between the most optimal arm's mean and the given one.
        :param arm: a given arm index
        :return: an absolute difference in means
        """
        ...

    def num_arms(self) -> int:
        ...
