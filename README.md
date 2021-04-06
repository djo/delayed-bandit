# Multi-armed bandit (MAB) problem under delayed feedback: numerical experiments

![Build](https://github.com/djo/delayed-bandit/workflows/Python%20application/badge.svg)

Provides the framework for numerical experiments to emulate the stochastic stationary environment with no delays
for the algorithms designed for the sequential resource allocation (under MAB problem) when rewards are delayed.

Created as a part of the paper "Multi-armed bandit problem in the delayed setting"
(*in print*) to provide the framework for numerical experiments.

Implementation of the framework to incorporate the existing algorithms into the stochastic environment under delays
is based on [Online Learning under Delayed Feedback](http://proceedings.mlr.press/v28/joulani13.html) paper.

Structure of the project and currently implemented algorithms:

||Files|
|-|-|
|Environments|[Protocol](delayed_bandit/environments/environment.py)|
||[Bernoulli MAB](delayed_bandit/environments/bernoulli_bandit.py)|
|Policies|[Protocol](delayed_bandit/policies/policy.py)|
||[Uniform Random](delayed_bandit/policies/uniform_random.py)|
||[Explore-First](delayed_bandit/policies/etc.py)|
||[Epsilon-Greedy](delayed_bandit/policies/epsilon_greedy.py)|
||[Thompson Sampling (Beta distribution)](delayed_bandit/policies/beta_thompson_sampling.py)|
|Experiments|[Bernoulli MAB under delayed feedback](delayed_bandit/experiments.py)|
|Tests|[Test module](delayed_bandit/test/)|

To run experiments on Bernoulli MAB see
```
python delayed_bandit/experiments.py --help
```

One might want to run a significant number of experiments and aggregate the result by removing outliers and averaging.
The sampling of delays might be fixated over the horizon.

![Bernoulli MAB under delayed feedback with Explore-First algorithm](bernoulli-mab-explore-then-commit.png)

![Comparison of algorithms in Bernoulli MAB with no delays](all-algorithms-no-delay.png)

![Comparison of algorithms in Bernoulli MAB under delay t=50](all-algorithms-delay-50.png)

![Comparison of algorithms in Bernoulli MAB under delay t=150](all-algorithms-delay-150.png)

### Development

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pytest
```

MIT License

Copyright (c) 2021 Andrii Dzhoha
