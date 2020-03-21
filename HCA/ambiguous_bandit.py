"""
Ambiguous Bandit
"""
import warnings
import numpy as np
import gym
from gym import spaces
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class AmbiguousBanditEnv(DiscreteEnv):
    """
    A three state MDP.  Start state is a bandit with two actions leading to one of two 
    absorbing/terminal states whose reward functions are similar (overlapping Gaussians).
    For each action, there is an epsilon probability of landing in the other state.

    Actions
    Num  Observation  Actions
    0    START        [0] a [1] a'
    1    HI           [0] a [1] a'
    2    LO           [0] a [1] a'
    """
    def __init__(self, epsilon=0.1,
                 means={'HI': 2, 'LO': 1},
                 stds={'HI': 1.5, 'LO': 1.5}):
        # map integer from discrete observation_space to name
        self._name_for_state = {0: 'START', 1: 'HI', 2: 'LO'}
        self.means = means
        self.stds = stds

        nS = len(self._name_for_state)
        nA = 2

        # initial state distribution (always starts at START)
        isd = np.array([1, 0, 0])

        # P is a dict of dict of lists where
        # P[s][a] == [(probability, nextstate, done),...]
        P = {o: {} for o in self._name_for_state}

        P[0][0] = [(1-epsilon, 1, True), (epsilon, 2, True)]
        P[0][1] = [(epsilon, 1, True), (1-epsilon, 2, True)]
        # 1 and 2 are terminal states, but define these actions to be
        # compatible with the step API in case user does not reset env.
        P[1][0] = [(1, 1, True)]
        P[1][1] = [(1, 1, True)]
        P[2][0] = [(1, 2, True)]
        P[2][1] = [(1, 2, True)]

        super().__init__(nS, nA, P, isd)

    def step(self, a):
        assert self.action_space.contains(a)

        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        if self.s == 0:
            # in START state
            r = self._bandit_reward(s)
        else:
            warnings.warn("In terminal state, should reset environment.")
            r = 0
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob": p})

    def _bandit_reward(self, s):
        """Sample reward from bandit

        Arg
        ---
        s : integer, the state

        Returns
        -------
        float, reward drawn from normal distribution for that state
        """
        state_name = self._name_for_state[s]
        return np.random.normal(self.means[state_name], self.stds[state_name])
