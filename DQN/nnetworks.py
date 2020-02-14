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
    def __init__(self, act_dim, w=84, h=84, num_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        self._convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=8, stride=4),
                                                      kernel_size=4,
                                                      stride=2),
                                      kernel_size=3,
                                      stride=1)
        self._convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=8, stride=4),
                                                      kernel_size=4,
                                                      stride=2),
                                      kernel_size=3,
                                      stride=1)
        linear_input_size = self._convw * self._convh * 64  # 7*7*64 for Atari

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, act_dim)

    def forward(self, obs):
        # Input has shape (batch size, width, height, num channels),
        # want shape (batch size, num channels, width, height).
        x = obs.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * self._convw * self._convh)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return torch.squeeze(q, -1)  # Ensure q has right shape.


class CNNCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        width = observation_space.shape[0]
        height = observation_space.shape[1]
        num_channels = observation_space.shape[2]
        # obs_num_channels = 4 for atari using frame_stack=True
        act_dim = action_space.n  # assumes Discrete space

        # build value function
        self.q = CNNQFunction(act_dim, width, height, num_channels)

    def act(self, obs):
        """Return an action (an integer)"""
        with torch.no_grad():
            obs = torch.unsqueeze(obs, 0)
            a = np.argmax(self.q(obs)).numpy()
            return a
        
