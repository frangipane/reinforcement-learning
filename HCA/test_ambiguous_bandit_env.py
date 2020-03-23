"""
Test expected returns from optimal, random, and worst case agent acxting
in AmbiguousBanditEnv.

A single test will fail ~5% of the time because of assertions checking the 
sample mean falls within 2 standard deviations of the population mean.
"""

import pytest
import numpy as np
from ambiguous_bandit import AmbiguousBanditEnv


@pytest.fixture(scope='module')
def bandit_kwargs():
    return dict(epsilon=0.1,
                means={'HI': 2, 'LO': 1},
                stds={'HI': 1.5, 'LO': 1.5},
                OHE_obs=False)


@pytest.fixture(scope='function')
def env(bandit_kwargs):
    return AmbiguousBanditEnv(**bandit_kwargs)


def run(env, n_episodes=1000, action=None):
    avg_reward = 0
    is_random_agent = action is None

    for ep in range(n_episodes):
        _ = env.reset()

        if is_random_agent:
            # random agent
            action = np.random.choice([0,1])
        _, r, _, _ = env.step(action)

        avg_reward += (r - avg_reward) / (ep+1)
    return avg_reward


## Assume same standard deviation for both HI and LO

def test_optimal_agent(bandit_kwargs, env):
    action = 0
    n_episodes = 1000
    observed_avg_reward = run(env, n_episodes=n_episodes, action=action)

    hi_mean_rew = bandit_kwargs['means']['HI']
    lo_mean_rew = bandit_kwargs['means']['LO']    
    std = bandit_kwargs['stds']['HI']
    epsilon = bandit_kwargs['epsilon']
    expected_avg_reward = (1-epsilon) * hi_mean_rew + epsilon * lo_mean_rew

    standard_error = std / np.sqrt(n_episodes)
    lo_range = expected_avg_reward - 2 * standard_error
    hi_range = expected_avg_reward + 2 * standard_error
    print(expected_avg_reward)
    print(observed_avg_reward)
    print(lo_range)
    print(hi_range)    
    assert observed_avg_reward > lo_range and \
        observed_avg_reward < hi_range


def test_random_agent(bandit_kwargs, env):
    action = None
    n_episodes = 1000
    observed_avg_reward = run(env, n_episodes=n_episodes, action=action)

    hi_mean_rew = bandit_kwargs['means']['HI']
    lo_mean_rew = bandit_kwargs['means']['LO']
    std = bandit_kwargs['stds']['HI']
    epsilon = bandit_kwargs['epsilon']
    expected_avg_reward = 0.5 * (hi_mean_rew + lo_mean_rew)

    standard_error = std / np.sqrt(n_episodes)
    lo_range = expected_avg_reward - 2 * standard_error
    hi_range = expected_avg_reward + 2 * standard_error
    print(expected_avg_reward)
    print(observed_avg_reward)
    print(lo_range)
    print(hi_range)    
    assert observed_avg_reward > lo_range and \
        observed_avg_reward < hi_range


def test_worst_agent(bandit_kwargs, env):
    action = 1
    n_episodes = 1000
    observed_avg_reward = run(env, n_episodes=n_episodes, action=action)

    hi_mean_rew = bandit_kwargs['means']['HI']
    lo_mean_rew = bandit_kwargs['means']['LO']    
    std = bandit_kwargs['stds']['HI']
    epsilon = bandit_kwargs['epsilon']
    expected_avg_reward = (epsilon) * hi_mean_rew + (1-epsilon) * lo_mean_rew

    standard_error = std / np.sqrt(n_episodes)
    lo_range = expected_avg_reward - 2 * standard_error
    hi_range = expected_avg_reward + 2 * standard_error
    print(expected_avg_reward)
    print(observed_avg_reward)
    print(lo_range)
    print(hi_range)    
    assert observed_avg_reward > lo_range and \
        observed_avg_reward < hi_range
