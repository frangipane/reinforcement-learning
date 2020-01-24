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


class StudentEnv(DiscreteLimitActionsEnv):
    """
    Student MDP from David Silver's Lecture 2.
    
    Actions (action_space):
      Type: Tuple([spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),])
      Num   Observation   Actions
      0     FACEBOOK      [0] facebook, [1] quit
      1     CLASS1        [0] facebook, [1] study
      2     CLASS2        [0] sleep, [1] study
      3     CLASS3        [0] pub, [1] study
      4     SLEEP         [0] sleep
    """   
    def __init__(self):
        # states / observations
        FACEBOOK = 0
        CLASS1 = 1
        CLASS2 = 2
        CLASS3 = 3
        SLEEP = 4  # terminal state
        observations = [FACEBOOK, CLASS1, CLASS2, CLASS3, SLEEP]

        nS = len(observations)

        # initial state distribution (uniform)
        isd = np.ones(nS) / nS

        # P is a dict of dict of lists, where
        #   P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {}
        for s in observations:
            P[s] = {}

        P[FACEBOOK][0] = [(1, FACEBOOK, -1, False)]
        P[FACEBOOK][1] = [(1, CLASS1, 0, False)]
        P[CLASS1][0] = [(1, FACEBOOK, -1, False)]
        P[CLASS1][1] = [(1, CLASS2, -2, False)]
        P[CLASS2][0] = [(1, SLEEP, 0, True)]
        P[CLASS2][1] = [(1, CLASS3, -2, False)]
        P[CLASS3][0] = [(0.2, CLASS1, 1, False), 
                        (0.4, CLASS2, 1, False), 
                        (0.4, CLASS3, 1, False)]
        P[CLASS3][1] = [(1, SLEEP, 10, True)]
        P[SLEEP][0] = [(1, SLEEP, 0, True)]
        
        vA = []
        for s in observations:
            vA.append(len(P[s]))
       
        super().__init__(nS, vA, P, isd)
