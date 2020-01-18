"""
OpenAI gym environment modeled after gym.envs.toy_text.discrete.DiscreteEnv,
but restricting the actions that can be taken per state.
"""
import numpy as np

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.toy_text.discrete import categorical_sample


class DiscreteLimitActionsEnv(Env):

    """
    Has the following members
    - nS: number of states
    - vA: vector of number of actions per state, same length as nS
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, vA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.vA = np.array(vA)

        assert (self.vA >= 0).all(), "Number of actions per state must be nonnegative."
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Tuple(tuple(spaces.Discrete(nA) for nA in self.vA))

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def step(self, a):
        if not self.action_space.spaces[self.s].contains(a):
            raise ValueError(
                f"Action must be < {self.action_space.spaces[self.s].n} in space {self.s}, attempted {a}"
            )
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.lastaction = (self.s, a)
        self.s = s
        return (s, r, d, {"prob" : p})
