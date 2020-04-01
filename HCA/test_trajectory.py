import pytest
import numpy as np

from tabular_vpg import Trajectory


@pytest.fixture(scope='module')
def steps():
    obs = [0]*4
    act = [0]*4
    rew = [1, 0.5, 0, -0.5]
    val = [0, 0.5, 1.0, 1.1]
    last_obs = 0
    last_val = 0
    return (obs, act, rew, val, last_obs, last_val)


@pytest.fixture(scope='module')
def traj_bootstrap(steps):
    """4 step episode
    states and actions don't matter
    """
    obs, act, rew, val, last_obs, last_val = steps
    tr = Trajectory(gamma=1.0, lam=1.0, bootstrap_n=2)
    for o, a, r, v in zip(obs, act, rew, val):
        tr.store(o, a, r, v)
    tr.finish_path(last_obs=last_obs, last_val=last_val)
    return tr


@pytest.fixture(scope='module')
def traj_MC(steps):
    """4 step episode
    states and actions don't matter
    """
    obs, act, rew, val, last_obs, last_val = steps
    tr = Trajectory(gamma=1.0, lam=1.0, bootstrap_n=None)
    for o, a, r, v in zip(obs, act, rew, val):
        tr.store(o, a, r, v)
    tr.finish_path(last_obs=last_obs, last_val=last_val)
    return tr


def test_traj_bootstrap(traj_bootstrap):
    rews = traj_bootstrap.rewards
    vals = traj_bootstrap.values
    vals.append(traj_bootstrap.last_val)

    n = traj_bootstrap.bootstrap_n
    g = traj_bootstrap.gamma

    # assumes trajectory has length 4
    assert len(rews) == 4
    expected_returns = [(g ** np.arange(n+1) * np.array(rews[0:n] + [vals[n]])).sum()] \
        + [(g ** np.arange(n+1) * np.array(rews[1:n+1] + [vals[n+1]])).sum()] \
        + [(g ** np.arange(n+1) * np.array(rews[2:n+2] + [vals[n+2]])).sum()] \
        + [(g ** np.arange(n) * np.append(np.array(rews[n+1]), vals[n+2])).sum()]  # truncated bootstrap
    assert expected_returns == traj_bootstrap.returns

    expected_adv = (np.array(expected_returns) - np.array(vals[:-1])).tolist()
    assert expected_adv == traj_bootstrap.advantage


def test_traj_MC(traj_MC):
    rews = traj_MC.rewards
    n = len(rews)
    g = traj_MC.gamma

    # assumes trajectory has length 4 and lamda=1 and last_val was 0 (episode ended)
    assert len(rews) == 4
    assert traj_MC.lam == 1.0
    assert traj_MC.last_val == 0.0
    expected_returns = [(g ** np.arange(n) * np.array(rews[0:])).sum()] \
                       + [(g ** np.arange(n-1) * np.array(rews[1:])).sum()] \
                       + [(g ** np.arange(n-2) * np.array(rews[2:])).sum()] \
                       + [(g ** np.arange(n-3) * np.array(rews[3:])).sum()]
    assert expected_returns == traj_MC.returns

    expected_adv = (np.array(expected_returns) - np.array(traj_MC.values)).tolist()
    assert expected_adv == traj_MC.advantage
