"""Following template laid out in Spinning up in RL, e.g.
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg
and
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
with some utils taken from
https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Deep%20Q%20Learning.ipynb

TODOs:
- implement logging
- implement test agent
- double check Monitor wrapper params (resume=True or False?)
- use more realistic hyperparameters, do rewards improve per episode?
- Atari environment-specific preprocessing for images, skip frames, concat inputs

"""
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn

import gym
from gym.wrappers import Monitor
# N.B. to use Monitor, need to have ffmpeg installed,
# e.g. on macOS: brew install ffmpeg

from nnetworks import *

#============================================================================

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    
    Copied from: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py#L11,
    modified action buffer for discrete action space.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)  # assumes discrete action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def dqn(env_fn, actor_critic=MLPCritic, replay_size=500,
        seed=0, steps_per_epoch=100, epochs=100,
        gamma=0.99, lr=0.00025, batch_size=32, start_steps=100, 
        update_after=100, update_every=5,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=15,
        target_update_every=1000,
        record_video_every=100):
    """
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method and a ``q`` module.
            The ``act`` method module should accept batches of
            observations as inputs, and ``q`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act`` and ``q`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        global_steps (int): Number of steps / frames for training (should be
            greater than update_after!)

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        lr (float): Learning rate.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates.

        epsilon_start (float): Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value

        epsilon_end (float): The final minimum value of epsilon after decaying is done

        epsilon_decay_steps (int): Number of steps to decay epsilon over

        target_update_every (int): Number of steps between updating target network
            parameters, i.e. resetting Q_hat to Q.

        record_video_every (int): Record a video every N episodes
    """
    env = env_fn()
    env = Monitor(
        env, 
        directory="/tmp/gym-results",
        resume=True,
        video_callable=lambda count: count % record_video_every == 0
    )
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n  # assumes Discrete space

    # Create critic module and network
    ac = actor_critic(env.observation_space, env.action_space)
    # Set target Q-network parameters theta_tar = theta
    target_q_network = deepcopy(ac.q)

    # Freeze target network w.r.t. optimizers
    for p in target_q_network.parameters():
        p.requires_grad = False    

    # function to compute Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        # Pick out q-values associated with / indexed by the action that was taken
        # for that observation: https://pytorch.org/docs/stable/torch.html#torch.gather.
        # Note index must be of type LongTensor.
        q = torch.gather(ac.q(o), dim=1, index=a.view(-1, 1).long())

        # Bellman backup for Q function
        with torch.no_grad():
            # Targets come from frozen target Q-network
            q_target = torch.max(target_q_network.q(o2), dim=1).values
            backup = r + (1 - d) * gamma * q_target
            
        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()
        # TODO: clip Bellman error b/w -1 and 1
        
        return loss_q

    # Set up optimizer for Q-function
    q_optimizer = torch.optim.RMSprop(ac.q.parameters(), lr=lr)

    # function to update parameters in Q
    def update(data):
        q_optimizer.zero_grad()
        loss = compute_loss_q(data)
        loss.backward()
        q_optimizer.step()

    def get_action(o):
        return ac.act(torch.as_tensor(o, dtype=torch.float32))

    def test_agent():
        pass

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    
    # main loop: collect experience in env

    # Initialize experience replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    o = env.reset()
    dqn_rewards_per_episode = []
    cum_reward = 0  # Track cumulative reward per episode

    total_steps = steps_per_epoch * epochs

    for t in range(total_steps):
        if t == total_steps - 1:
            print('done')
            print('epsilon ', epsilon)
        
        # epsilon for this time step
        epsilon = epsilons[min(t, epsilon_decay_steps-1)]

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration.  Afterwards,
        # follow an epsilon-greedy approach using the learned Q network.
        if t > start_steps:
            # epsilon greedy
            if np.random.sample() < epsilon:
                a = env.action_space.sample()
            else:
                a = get_action(o)
        else:
            a = env.action_space.sample()
            
        # Step the env
        o2, r, d, _ = env.step(a)
        cum_reward += r
        
        # Store transition to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Update the most recent observation.
        o = o2
        
        # End of episode handling
        if d:
            dqn_rewards_per_episode.append(cum_reward)
            o = env.reset()
            cum_reward = 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            minibatch = replay_buffer.sample_batch(batch_size)
            update(data=minibatch)
            
        # Refresh target Q network
        if t % target_update_every == 0:
            target_q_network.load_state_dict(ac.q.state_dict())
            
    env.close()
    return dqn_rewards_per_episode


if __name__ == '__main__':
    # TODO: WIP, should take in config
    # for dqn args
    import matplotlib
    import matplotlib.pyplot as plt
    
    rewards = dqn(lambda : gym.make('CartPole-v1'))
    plt.plot(rewards)
