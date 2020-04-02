import numpy as np
from abc import ABC, abstractmethod


__all__ = ['TabularReturnHCA', 'TabularStateHCA', 'TabularVPGActorCritic']


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

    def __str__(self):
        return self.__name__

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


#####################################################################
# HCA algo implemen ted in compute_errors copied from
# https://github.com/hca-neurips2019/hca/blob/master/hca_classes.py
#####################################################################

class TabularReturnHCA(BaseTabularActorCritic):
    """
    Discrete observation and action space
    """
    def __init__(self, obs_dim, act_dim, return_bins,
                 pi_lr=0.1, vf_lr=0.1, h_lr=0.1):
        """return_bins is a 1-dimensional np.array
        """
        self._return_bins = return_bins
        num_bins = len(return_bins)
        self.logits_h = np.zeros((act_dim, obs_dim, num_bins))
        self.h_lr = h_lr
        super().__init__(obs_dim, act_dim, pi_lr, vf_lr)

    @property
    def h(self):
        Z = np.exp(self.logits_h).sum(axis=0, keepdims=True)
        h = np.exp(self.logits_h) / Z
        return h

    def compute_errors(self, traj):
        T = len(traj.states)
        dlogits_pi = np.zeros_like(self.pi)
        dV = np.zeros_like(self.V)
        dlogits_h = np.zeros_like(self.h)

        for i in range(T):
            x_s, a_s, G = traj.states[i], traj.actions[i], traj.returns[i]
            G_bin_ind = (np.abs(self._return_bins - G)).argmin()
            hca_factor = (1. - self.pi[x_s, :] / self.h[:, x_s, G_bin_ind])
            G_hca = G * hca_factor

            dlogits_pi[x_s, a_s] += G_hca[a_s]
            dlogits_pi[x_s] -= self.pi[x_s] * G_hca[a_s]
            dV[x_s] += (G - self.V[x_s])
            dlogits_h[a_s, x_s, G_bin_ind] += 1
            dlogits_h[:, x_s, G_bin_ind] -= self.h[:, x_s, G_bin_ind]
        return dlogits_pi, dV, dlogits_h

    def update(self, traj):
        dlogits_pi, dV, dlogits_h = self.compute_errors(traj)
        self.logits_pi += self.pi_lr * dlogits_pi
        self.V += self.vf_lr * dV
        self.logits_h += self.h_lr * dlogits_h


class TabularStateHCA(BaseTabularActorCritic):
    def __init__(self, obs_dim, act_dim,
                 pi_lr=0.1, vf_lr=0.1, h_lr=0.1):
        self.logits_h = np.zeros((act_dim, obs_dim, obs_dim))  # double check this
        self.h_lr = h_lr
        super().__init__(obs_dim, act_dim, pi_lr, vf_lr)

    @property
    def h(self):
        Z = np.exp(self.logits_h).sum(axis=0, keepdims=True)
        h = np.exp(self.logits_h) / Z
        return h

    def compute_errors(self, traj):
        T = len(traj.states)
        dlogits_pi = np.zeros_like(self.pi)
        dV = np.zeros_like(self.V)
        dlogits_h = np.zeros_like(self.h)

        for i in range(T):
            x_s, a_s, G = traj.states[i], traj.actions[i], traj.returns[i]
            G_hca = np.zeros_like(self._actions, dtype=float)

            for j in range(i, T+1):
                if j == T:
                    x_t, r = traj.last_obs, traj.last_val
                else:
                    x_t, r = traj.states[j], traj.rewards[j]
                hca_factor = self.h[:, x_s, x_t].T - self.pi[x_s, :]
                G_hca += traj.gamma**(j-i) * r * hca_factor

                dlogits_h[a_s, x_s, x_t] += 1
                dlogits_h[:, x_s, x_t] -= self.h[:, x_s, x_t]

            for a in self._actions:
                dlogits_pi[x_s, a] += G_hca[a]
                dlogits_pi[x_s] -= self.pi[x_s] * G_hca[a]
            dV[x_s] += (G - self.V[x_s])

        return dlogits_pi, dV, dlogits_h

    def update(self, traj):
        dlogits_pi, dV, dlogits_h = self.compute_errors(traj)
        self.logits_pi += self.pi_lr * dlogits_pi
        self.V += self.vf_lr * dV
        self.logits_h += self.h_lr * dlogits_h
