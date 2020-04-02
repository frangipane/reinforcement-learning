import numpy as np
import warnings
from spinup.utils.logx import EpochLogger

import hca.core as core
import hca.tabular_actor_critic as tabular_actor_critic
import wandb


class Trajectory:
    """
    Size depends on length of trajectory, assuming
    episodic environment of small size for toy problems.
    """
    def __init__(self, gamma=1.0, lam=0.95, bootstrap_n=None):
        self.gamma = gamma
        self.lam = lam
        self.bootstrap_n = bootstrap_n
        if bootstrap_n is not None and lam != 1.0:
            warnings.warn(
                f"Using bootstrap_n of {bootstrap_n}; lambda {lam} will be ignored")
        self.reset()

    def store(self, obs, act, rew, val):
        self.states.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.values.append(val)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantage = []
        self.last_obs = None
        self.last_val = None

    def finish_path(self, last_obs, last_val):
        self.last_obs = last_obs
        self.last_val = last_val

        deltas = np.array(self.rewards) \
                 + self.gamma * np.append(np.array(self.values[1:]), last_val) \
                 - np.array(self.values)

        if self.bootstrap_n is None:
            # GAE-lambda advantage
            self.advantage = core.discount_cumsum(deltas, self.gamma * self.lam).tolist()
            self.returns = core.discount_cumsum(np.array(self.rewards), self.gamma).tolist()  # rewards-to-go
        else:
            # see eq14 in High-Dimensional Continuous Control using Generalized Advantage Estimation
            # https://arxiv.org/abs/1506.02438
            n_states = len(self.states)
            adv = np.zeros(n_states)
            # TODO: vectorize this
            deltas = np.append(deltas, np.zeros(self.bootstrap_n))
            for t in range(n_states):
                adv[t] = (self.gamma ** np.arange(self.bootstrap_n) \
                         * deltas[t:t+self.bootstrap_n]).sum()
            self.returns = (adv + np.array(self.values)).tolist()
            self.advantage = adv.tolist()


def vpg(env_fn, actor_critic=tabular_actor_critic.TabularVPGActorCritic,
        n_episodes=100, env_kwargs={}, logger_kwargs={}, ac_kwargs={},
        n_test_episodes=100, gamma=0.99, lam=0.95, bootstrap_n=3):
    """
    Environment has discrete observation and action spaces, both
    low dimensional so policy and value functions can be stored
    in table.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic : The constructor method for an actor critic class
            with an ``act`` method, and attributes ``pi`` and ``v``.

        n_episodes (int): Number of episodes/rollouts of interaction (equivalent
            to number of policy updates) to perform.

        bootstrap_n (int) : (optional) Number of reward steps to use with a bootstrapped
            approximate Value function.  If None, use GAE-lambda advantage estimation.
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env = env_fn(**env_kwargs)
    test_env = env_fn(**env_kwargs)

    obs_dim = env.observation_space.n
    act_dim = env.action_space.n

    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)

    def test_agent():
        o, test_ep_ret, test_ep_len = test_env.reset(), 0, 0

        episode = 0
        while episode < n_test_episodes:
            a, _ = ac.step(o)
            o2, r, d, _ = test_env.step(a)
            test_ep_ret += r
            test_ep_len += 1

            o = o2

            if d is True:
                logger.store(TestEpRet=test_ep_ret)
                logger.store(TestEpLen=test_ep_len)
                episode += 1
                o, test_ep_ret, test_ep_len = test_env.reset(), 0, 0

    traj = Trajectory(gamma, lam, bootstrap_n)

    # Run test agent before any training happens
    episode = 0
    test_agent()
    print('Mean test returns from random agent:', np.mean(logger.epoch_dict['TestEpRet']), flush=True)
    logger.log_tabular('Epoch', episode)
    logger.log_tabular('TestEpRet', with_min_and_max=True)
    logger.log_tabular('TestEpLen', with_min_and_max=True)
    # Hack logger values for compatibility with main logging header keys
    logger.log_tabular('EpRet', 0)
    logger.log_tabular('EpLen', 0)
    logger.log_tabular('AverageVVals', 0)
    logger.log_tabular('MaxVVals', 0)
    logger.log_tabular('MinVVals', 0)
    logger.log_tabular('StdVVals', 0)
    logger.log_tabular('TotalEnvInteracts', 0)
    wandb.log(logger.log_current_row, step=episode)
    logger.dump_tabular()

    episode += 1
    o, ep_ret, ep_len = env.reset(), 0, 0
    total_env_interacts = 0

    while episode < n_episodes:
        a, v = ac.step(o)
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        total_env_interacts += 1

        traj.store(o, a, r, v)
        logger.store(VVals=v)

        o = o2

        if d is True:
            traj.finish_path(last_obs=o, last_val=0)
            ac.update(traj)
            test_agent()

            logger.log_tabular('Epoch', episode)
            logger.log_tabular('EpRet', ep_ret)
            logger.log_tabular('EpLen', ep_len)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', total_env_interacts)
            wandb.log(logger.log_current_row, step=episode)
            logger.dump_tabular()

            traj.reset()
            episode += 1
            o, ep_ret, ep_len = env.reset(), 0, 0

    print('pi', ac.pi, flush=True)
    print('logits_pi', ac.logits_pi, flush=True)
    print('value', ac.V, flush=True)
    if isinstance(ac, tabular_actor_critic.TabularReturnHCA) or isinstance(ac, tabular_actor_critic.TabularStateHCA):
        print('h', ac.h, flush=True)


if __name__ == '__main__':
    import shortcut_env
    env_kwargs={'OHE_obs': False, 'random_start': True, 'n': 5}
    vpg(shortcut_env.ShortcutEnv, env_kwargs=env_kwargs, n_episodes=10)
