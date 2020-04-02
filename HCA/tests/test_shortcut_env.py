import pytest
import numpy as np
from hca.envs.shortcut_env import ShortcutEnv


def run(env, n_episodes=100, action=None):
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


def test_always_shortcut():
    action = 0
    n = 5
    final_state = n - 1
    num_trials = 20
    env = ShortcutEnv(n=n, random_start=True, OHE_obs=False)

    for i in range(num_trials):
        o = env.reset()
        if o != final_state:
            o2, r, _, _ = env.step(action)
            assert r == 0.0
            assert o2 == n-1


def test_run_random_agent():
    """Just a functional test"""
    env = ShortcutEnv(n=5, random_start=True, OHE_obs=False)
    returns = run(env, n_episodes=500)
