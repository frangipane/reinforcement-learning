import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



#============================================================================
# Use this for non-image inputs, e.g. CartPole-v1 gym environment

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
            a = torch.argmax(self.q(obs)).numpy()
            return a


#============================================================================
# Use this for image inputs, e.g. Atari
# (WIP: untested)

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
        
