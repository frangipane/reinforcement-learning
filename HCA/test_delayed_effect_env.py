"""
Test expected returns from delayed effect env
"""

import pytest
import numpy as np
from delayed_effect_env import DelayedEffectEnv


@pytest.fixture(scope='module')
def env_kwargs():
    return dict(n=3, final_reward=1.0, OHE_obs=False)


@pytest.fixture(scope='function')
def env(env_kwargs):
    return DelayedEffectEnv(**env_kwargs)


def run(env, first_act=None):
    first_step = True
    d = False

    _ = env.reset()
    while not d:
        if first_step is False:
            # just take random action after first action
            o, r, d, _ = env.step(np.random.choice([0,1]))
            print(o,r,d)
        else:
            env.step(first_act)
            first_step = False
    return (o, r)


def test_optimal_agent(env, env_kwargs):
    first_act = 0
    final_obs, final_rew = run(env, first_act=first_act)

    expected_final_state = env_kwargs['n'] + 1
    expected_final_rew = env_kwargs['final_reward']
    assert final_obs == expected_final_state and final_rew == expected_final_rew


def test_worst_agent(env, env_kwargs):
    first_act = 1
    final_obs, final_rew = run(env, first_act=first_act)

    expected_final_state = env_kwargs['n'] + 2
    expected_final_rew = -1 * env_kwargs['final_reward']
    assert final_obs == expected_final_state and final_rew == expected_final_rew
    
