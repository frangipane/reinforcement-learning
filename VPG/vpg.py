import os
import time
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import wandb
from spinup.utils.logx import EpochLogger


pylogger = logging.getLogger(__name__)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pylogger.info(f"\n ******** Number of GPUs: {torch.cuda.device_count()}")


#============================================================================
# Use this for non-image inputs, e.g. CartPole-v1 gym environment

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPValueFunction(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.nn = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        """
        Args
        ----
        obs : Tensor, of observations, of shape (batch,)

        Returns
        -------
        v : Tensor, contains value estimates for the observations.
            shape (batch,)
        """
        return torch.squeeze(self.nn(obs), -1)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.nn = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        return torch.distributions.Categorical(logits=self.nn(obs))

    def forward(self, obs, actions=None):
        """
        Args
        ----
        obs : Tensor, of observations, of shape (batch,)
        actions : Tensor

        Returns
        -------
        pi : Torch Distribution object, containing batch of distributions
             describing the policy for the obs
        logp_a : (optional) Tensor containing the log probability of the actions.
                 None if no actions are given as input.
        """
        pi = self._distribution(obs)
        logp_a = pi.log_prob(actions)
        return logp_a


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n  # assumes Discrete space
        
        self.pi = MLPPolicy(obs_dim, act_dim, hidden_sizes, activation)
        self.v = MLPValueFunction(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        """
        Args
        ----
        obs : Tensor, of observations, of shape (batch, **obs feature dims)

        Returns
        -------
        a : numpy array of actions for each observation,
            shape (batch, act_dim)
        v : numpy array of value estimates, 
            shape (batch,)
        logp_a : numpy array of log probs for actions in a,
             shape (batch,)
        """
        with torch.no_grad():
            v = self.v(obs)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
        return a, v, logp_a


#============================================================================

class VPGBuffer:

    def __init__(self, size, gamma, obs_dim, act_dim):
        self.size = size
        self.gamma = gamma
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.int32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.logpa_buf = np.zeros(size, dtype=np.float32)
        self.advantage_buf = np.zeros(size, dtype=np.float32)
        self.rew2go_buf = np.zeros(size, dtype=np.float32)
        self.start_ptr, self.end_ptr = 0, 0

    def store(self, o, a, v, r, logpa):
        if self.end_ptr == self.size:
            raise Exception("Buffer is full")
        self.obs_buf[self.end_ptr] = o
        self.act_buf[self.end_ptr] = a
        self.val_buf[self.end_ptr] = v
        self.rew_buf[self.end_ptr] = r
        self.logpa_buf[self.end_ptr] = logpa
        self.end_ptr += 1

    def finish_trajectory(self):
        # Calculate rewards to go
        traj_slice = slice(self.start_ptr, self.end_ptr)
        self.rew2go_buf[traj_slice] = self._calculate_rewards_to_go(self.rew_buf[traj_slice],
                                                                    self.gamma)

        # calculate advantage using 1-step estimate of Q: r + gamma*V(s_t+1) - V(s_t)
        self._calculate_advantage()
        self.start_ptr = self.end_ptr

    def _calculate_advantage(self):
        # TODO: normalize advantages?
        """1-step estimate of Advantage"""
        i = self.start_ptr
        j = self.end_ptr
        self.advantage_buf[i:j-1] = self.rew_buf[i:j-1] \
                                    + self.gamma * self.val_buf[i+1:j] - self.val_buf[i:j-1]
        self.advantage_buf[j-1] = self.rew_buf[j-1] - self.val_buf[j-1]

    @staticmethod
    def _calculate_rewards_to_go(rewards, gamma):
        """
        For every timestep in rewards, calculate the discounted cumulative
        rewards after and including the reward for that step.

        Args
        ----
        rewards : np.array, shape (1, num steps in trajectory)

        Returns
        -------
        rew2go : np.array, same shape as rewards
        """
        steps_in_traj = len(rewards)
        rew2go = np.zeros(steps_in_traj, dtype=np.float32)
        for i in reversed(range(steps_in_traj)):
            if i == steps_in_traj - 1:
                rew2go[i] = rewards[i]
            else:
                rew2go[i] = gamma * rew2go[i+1] + rewards[i]
        return rew2go


def test_calculate_rewards_to_go():
    rewards = np.array([4, 3, 0, 1], dtype=np.float32)
    #expected_rewards2go = np.array([8, 4, 1, 1], dtype=np.float32)
    expected_rewards2go = np.array([5.625, 3.25, 0.5, 1], dtype=np.float32)
    buffer = VPGBuffer(1, 1, 1, 1)  # nonsense inputs ok
    obs_rewards2go = buffer._calculate_rewards_to_go(rewards, gamma=0.5)
    assert np.array_equal(obs_rewards2go, expected_rewards2go)

#============================================================================


def vpg(env, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10):
    """
    Vanilla Policy Gradient 
    (with GAE 0 for advantage estimation)
    Args:
        env : An environment that satisfies the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n  # assumes Discrete space

    ac = actor_critic(env.observation_space, env.action_space)
    ac.to(device)

    # buffer size equals number of steps in an epoch
    buff = VPGBuffer(steps_per_epoch, gamma, obs_dim, act_dim)

    def compute_loss_pi(data):
        obs = torch.as_tensor(data.obs_buf, dtype=torch.float32, device=device)
        act = torch.as_tensor(data.act_buf, dtype=torch.int32, device=device)      
        adv = torch.as_tensor(data.advantage_buf, dtype=torch.float32, device=device)
        logpa = ac.pi(obs, act)
        return -1 * (logpa * adv).mean()

    def compute_loss_v(data):
        obs = torch.as_tensor(data.obs_buf, dtype=torch.float32, device=device)
        rew2go = torch.as_tensor(data.rew2go_buf, dtype=torch.float32, device=device)
        values = ac.v(obs)
        return F.mse_loss(values, rew2go)

    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    v_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update_pi(data):
        pi_optimizer.zero_grad()
        pi_loss = compute_loss_pi(data)
        pi_loss.backward()
        pi_optimizer.step()

        logger.store(LossPi=pi_loss.item())
        #TODO: log policy entropy

    def update_v(data):
        for s in range(train_v_iters):
            v_optimizer.zero_grad()
            v_loss = compute_loss_v(data)
            v_loss.backward()
            v_optimizer.step()

            logger.store(LossV=v_loss.item())

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    t = 0  # total environment interactions

    # Update policy once per epoch
    for epoch in range(epochs):
        for t_epoch in range(steps_per_epoch):
            t += 1
            a, v, logpa = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
            o2, r, d, info = env.step(a.cpu().numpy())
            buff.store(o, a, v, r, logpa)

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==max_ep_len else d

            o = o2

            # If trajectory is finished, calculate rewards to go,
            # then calculate the Advantage.
            if d is True or (ep_len == max_ep_len) or (t_epoch + 1 == steps_per_epoch):
                buff.finish_trajectory()
                logger.store(EpRet=ep_ret,
                             EpLen=ep_len, )

                o, ep_ret, ep_len = env.reset(), 0, 0

            # Calculate policy gradient when we've collected t_epoch time steps.
            if t_epoch + 1 == steps_per_epoch:
                pylogger.debug('*** epoch ***', epoch)
                pylogger.debug('*** t_epoch ***', t_epoch)
                pylogger.debug('values', buff.val_buf)
                pylogger.debug('rewards', buff.rew_buf)
                pylogger.debug('rew2go', buff.rew2go_buf)
                pylogger.debug('advantage', buff.advantage_buf)

                # Update the policy using policy gradient
                update_pi(buff)

                # Re-fit the value function on the MSE.  Note, this is
                # gradient descent starting from the previous parameters.
                update_v(buff)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs):
            logger.save_state({'env': env}, None)  # note, this includes full model pickle

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('TotalEnvInteracts', t)
        logger.log_tabular('Time', time.time() - start_time)
        if hasattr(env, 'episode_id'):
            logger.log_tabular('EpisodeId', env.episode_id)

        # If a quantity has not been calculated/stored yet, do not log it.  This can
        # happen, e.g. if NN update length or episode length exceeds num steps in epoch.
        to_log = [{'key': 'LossV', 'average_only': True},
                  {'key': 'LossPi', 'average_only': True},
                  {'key': 'EpRet', 'with_min_and_max': True},
                  {'key': 'EpLen', 'average_only': True},
                  {'key': 'RawRet', 'with_min_and_max': True},
                  {'key': 'RawLen', 'average_only': True}]

        for log_tabular_kwargs in to_log:
            key = log_tabular_kwargs['key']
            if key in logger.epoch_dict and len(logger.epoch_dict[key]) > 0:
                logger.log_tabular(**log_tabular_kwargs)

        wandb.log(logger.log_current_row, step=epoch)
        logger.dump_tabular()

        # reset buffer
        buff = VPGBuffer(steps_per_epoch, gamma, obs_dim, act_dim)

    # Save final model as a state dict
    state = {
        'epoch': epoch,
        'pi_state_dict': ac.pi.state_dict(),
        'v_state_dict': ac.v.state_dict(),
        'pi_optimizer': pi_optimizer.state_dict(),
        'v_optimizer': v_optimizer.state_dict(),
    }
    # hack for wandb: should output the model in the wandb.run.dir to avoid
    # problems syncing the model in the cloud with wandb's files
    state_fname = os.path.join(logger_kwargs['output_dir'], f"state_dict.pt")
    torch.save(state, state_fname)
    wandb.save(state_fname)
    pylogger.info(f"Saved state dict to {state_fname}")
    env.close()

config = dict(
    seed=0, 
    steps_per_epoch=4000,  # this is also the buffer size
    epochs=2000,
    gamma=0.99,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_v_iters=80,
    max_ep_len=500,
    save_freq=200
)

wandb.init(project="vpg", config=config, tags=['CartPole-v1'])

addl_config = dict(
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(),
    logger_kwargs={'exp_name': 'vpg', 'output_dir': wandb.run.dir}
                   #'output_dir': f'/tmp/vpg/{time.time()}'},
)
    
if __name__ == '__main__':
    wandb.save("state_dict.pt")
    env = gym.make('CartPole-v1')
    vpg(env, **config, **addl_config)
