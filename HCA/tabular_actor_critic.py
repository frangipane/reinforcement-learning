import numpy as np
from abc import ABC, abstractmethod


class BaseTabularActorCritic(ABC):
    """
    Discrete observation and action space
    """
    def __init__(self, obs_dim, act_dim, pi_lr=0.1, vf_lr=0.1):
        self._actions = np.array(range(act_dim))
        self.logits_pi = np.zeros((obs_dim, act_dim))
        self.V = np.zeros(obs_dim)
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr

    @property
    def pi(self):
        Z = np.exp(self.logits_pi).sum(axis=1, keepdims=True)
        pi = np.exp(self.logits_pi) / Z
        return pi

    def step(self, obs):
        a =  np.random.choice(self._actions, p=self.pi[obs, :])
        v = self.V[obs]
        return a, v

    @abstractmethod
    def update(self, traj):
        """Update attributes, e.g. logits and V"""
        pass


class TabularVPGActorCritic(BaseTabularActorCritic):
    """
    Discrete observation and action space
    """
    def __init__(self, obs_dim, act_dim, pi_lr=0.1, vf_lr=0.1):
        super().__init__(obs_dim, act_dim, pi_lr, vf_lr)

    def compute_errors(self, traj):
        T = len(traj.states)
        dlogits_pi = np.zeros_like(self.pi)
        dV = np.zeros_like(self.V)

        for i in range(T):
            x_s, a_s, ret, adv = traj.states[i], traj.actions[i], traj.returns[i], traj.advantage[i]
            dlogits_pi[x_s, a_s] += adv
            dlogits_pi[x_s] -= self.pi[x_s] * adv
            dV[x_s] += (ret - self.V[x_s])
        return dlogits_pi, dV

    def update(self, traj):
        dlogits_pi, dV = self.compute_errors(traj)
        self.logits_pi += self.pi_lr * dlogits_pi
        self.V += self.vf_lr * dV
