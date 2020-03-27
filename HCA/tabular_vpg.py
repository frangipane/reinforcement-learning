import numpy as np
from spinup.utils.logx import EpochLogger

import core
import wandb


class TabularVPGActorCritic:
    """
    Discrete observation and action space
    """
    def __init__(self, obs_dim, act_dim, pi_lr=0.1, vf_lr=0.1):
        self._actions = np.array(range(act_dim))
        self.logits = np.zeros((obs_dim, act_dim))
        self.V = np.zeros(obs_dim)
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr

    @property
    def pi(self):
        Z = np.exp(self.logits).sum(axis=1, keepdims=True)
        pi = np.exp(self.logits) / Z
        return pi

    def step(self, obs):
        a =  np.random.choice(self._actions, p=self.pi[obs, :])
        v = self.V[obs]
        return a, v

    def compute_errors(self, traj):
        T = len(traj.states)
        dlogits = np.zeros_like(self.pi)
        dV = np.zeros_like(self.V)

        for i in range(T):
            x_s, a_s, ret, adv = traj.states[i], traj.actions[i], traj.returns[i], traj.advantage[i]
            dlogits[x_s, a_s] += adv
            dlogits[x_s] -= self.pi[x_s] * adv
            dV[x_s] += (ret - self.V[x_s])
        return dlogits, dV

    def update(self, traj):
        dlogits, dV = self.compute_errors(traj)
        self.logits += self.pi_lr * dlogits
        self.V += self.vf_lr * dV


class Trajectory:
    """
    Size depends on length of trajectory, assuming
    episodic environment of small size for toy problems.
    """
    def __init__(self, gamma=1.0, lam=0.95):
        self.gamma = gamma
        self.lam = lam
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

    def finish_path(self):
        # GAE-lambda advantage
        deltas = np.array(self.rewards) \
                 + self.gamma * np.append(np.array(self.values[1:]), 0.) \
                 - np.array(self.values)
        self.advantage = core.discount_cumsum(deltas, self.gamma * self.lam)

        self.returns = core.discount_cumsum(np.array(self.rewards), self.gamma).tolist()  # rewards-to-go


def vpg(env_fn, actor_critic=TabularVPGActorCritic,
        n_episodes=100, env_kwargs={}, logger_kwargs={}, ac_kwargs={},
        n_test_episodes=100, gamma=0.99, lam=0.95):
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

    traj = Trajectory(gamma, lam)

    o, ep_ret, ep_len = env.reset(), 0, 0

    total_env_interacts = 0
    episode = 0

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
            traj.finish_path()
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
    print('logits', ac.logits, flush=True)
    print('value', ac.V, flush=True)


if __name__ == '__main__':
    import shortcut_env
    env_kwargs={'OHE_obs': False, 'random_start': True, 'n': 5}
    vpg(shortcut_env.ShortcutEnv, env_kwargs=env_kwargs, n_episodes=10)
