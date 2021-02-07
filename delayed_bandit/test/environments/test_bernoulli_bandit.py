import pytest
from numpy.random import default_rng, Generator, PCG64

from delayed_bandit.environments.bernoulli_bandit import BernoulliBandit


class TestBernoulliBandit:
    def test_pull_with_certain_parameters(self):
        with_certainty = BernoulliBandit(means=[0.0, 1.0], rng=default_rng())
        assert with_certainty.pull(arm=0) == 0
        assert with_certainty.pull(arm=1) == 1

    def test_pull_with_fixed_seed(self):
        b = BernoulliBandit(means=[0.5], rng=Generator(PCG64(42)))
        assert b.pull(0) == 1
        assert b.pull(0) == 0

    def test_optimal_arm(self):
        b = BernoulliBandit(means=[0.64, 0.65, 0.65, 0.1], rng=default_rng())
        assert b.optimal_arm() == (1, 0.65)

    def test_suboptimality_gap(self):
        b = BernoulliBandit(means=[0.0, 0.65, 0.5], rng=default_rng())
        assert pytest.approx(b.suboptimality_gap(0)) == 0.65
        assert pytest.approx(b.suboptimality_gap(1)) == 0.0
        assert pytest.approx(b.suboptimality_gap(2)) == 0.15
