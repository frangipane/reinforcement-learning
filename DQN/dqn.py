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

"""
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from gym.wrappers import Monitor
# N.B. to use Monitor, need to have ffmpeg installed, e.g. on macOS: brew install ffmpeg

#============================================================================
# Use this for non-image inputs, e.g. cartpole

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        q = self.q(obs)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


# TODO: Consider removing this class entirely and handling `act` in dqn logic instead.
class MLPCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n  # assumes Discrete space

        # build value function
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        """Return an action (an integer)"""
        with torch.no_grad():
            a = np.argmax(self.q(obs)).numpy()
            return a


#============================================================================
# Use this for image inputs, e.g. Atari
# (untested)

class CNNQFunction(nn.Module):
    """
    Q-Value function approximator, a convolutional neural network.

    Use this network for both the Q-network (the approximator), and the 
    target network (Q-hat, that has frozen parameters for a certain 
    number of data gathering steps.
    
    The CNN is has 3 convolutional layers and 2 fully connected layers.  
    Each layer is followed by a ReLU.  There is a single output
    for each valid action.
    
    Args
    ----
    nA: number of actions
    """
    def __init__(self, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, act_dim)

    def forward(self, obs):
        x = obs.view(-1, 84, 84, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return torch.squeeze(q, -1)  # Ensure q has right shape.


class CNNCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        # TODO: do I need observation_space?
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n  # assumes Discrete space

        # build value function
        self.q = CNNQFunction(act_dim)

    def act(self, obs):
        """Return an action (an integer)"""
        with torch.no_grad():
            a = np.argmax(self.q(obs)).numpy()
            return a
        
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
        self.act_buf = np.zeros(size, dtype=np.float32)  # assumes discrete action
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
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def dqn(env_fn, actor_critic=MLPCritic, global_steps=10000, replay_size=500, 
        seed=0,
        gamma=0.99, lr=0.00025, batch_size=32, start_steps=100, 
        update_after=100, update_every=5,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=15,
        target_update_every=1000,
        record_video_every=100):
    
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

    # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    
    # main loop: collect experience in env

    # Initialize experience replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    o = env.reset()
    dqn_rewards_per_episode = []
    cum_reward = 0  # Track cumulative reward per episode

    for t in range(global_steps):        
        if t == global_steps - 1:
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


# TO RUN
# rewards = dqn(lambda : gym.make('CartPole-v1'))

# # plot rewads in jupyter
# import matplotlib
# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.plot(rewards)
