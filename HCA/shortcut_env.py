"""
Shortcut environment -- Hindsight Credit Assignment Fig 2 (Left)
"""
import numpy as np
import torch
import gym
from gym.envs.toy_text.discrete import DiscreteEnv


class ShortcutEnv(DiscreteEnv):
    """
    A chain of length n with a rewarding final state.  At each step, one action takes
    a shortcut and directly transitions to the final state, while the other action
    continues on the longer path.

    There is a per-step penalty of -1 and a final reward of 1.
    There is also a 0.1 chance that the agent transitions to the absorbing state directly.

    Num  Observation  Actions
    0    0            [0] shortcut [1] long
    1    1            [0] shortcut [1] long
    ...
    N-1  N-1          [0] shortcut [1] long  # terminal

    Transition probabilities (for i != N-2):
    P(s_{N-1} | s_{i}, shortcut) = 1.0
    P(s_{N-1} | s_{i}, long) = epsilon
    P(s_{i+1} | s_{i}, long) = 1 - epsilon

    for the next to last state (i = N-2):
    P(s_{N-1} | s_{N-2}, shortcut) = 1.0
    P(s_{N-1} | s_{N-2}, long) = 1.0

    Args
    ----
    n : int
        number of states

    epsilon : float
        probability of transitioning to final state when taking the long action

    step_penalty : float
        reward for each step that does not end in final state

    final_reward : float
        reward for reaching final state

    random_start : bool
        if True, uniformly sample from states for starting state, else always start from first state

    OHE_obs : bool
        if False, obs is an integer >= 0, the default from the discrete space. 
        if True, convert observation to a one hot encoded torch tensor of length n,
        the number of states in the MDP
    """
    def __init__(self, n=5,
                 epsilon=0.1,
                 step_penalty=-1.0,
                 final_reward=1.0,
                 random_start=True,
                 OHE_obs=True):
        nS = n
        nA = 2
        final_state = n-1
        a_shortcut = 0
        a_long = 1
        self.nS = n
        self._OHE_obs = OHE_obs

        if random_start:
            isd = np.ones(nS) / (nS - 1)
            isd[nS-1] = 0.  # don't ever start in terminal state
        else:
            # always start at first state
            isd = np.zeros(nS)
            isd[0] = 1.0

        # P is a dict of dict of lists where
        # P[s][a] == [(probability, nextstate, reward, done),...]
        P = {o: {} for o in range(n)}

        for o in range(nS-2):
            P[o][a_shortcut] = [(1.0, final_state, final_reward + step_penalty, True)]
            P[o][a_long] = [(epsilon, final_state, final_reward + step_penalty, True),
                            (1-epsilon, o + 1, step_penalty, False)]

        # penultimate state
        P[final_state - 1][a_shortcut] = [(1.0, final_state, final_reward + step_penalty, True)]
        P[final_state - 1][a_long] = [(1.0, final_state, final_reward + step_penalty, True)]

        # final state
        P[final_state][a_shortcut] = [(1.0, final_state, 0, True)]
        P[final_state][a_long] = [(1.0, final_state, 0, True)]        

        super().__init__(nS, nA, P, isd)

    def step(self, a):
        s, r, d, info = super().step(int(a))
        if self._OHE_obs:
            s = torch.nn.functional.one_hot(torch.as_tensor(s), self.nS)
        return (s, r, d, info)

    def reset(self):
        s = super().reset()
        if self._OHE_obs:
            s = torch.nn.functional.one_hot(torch.as_tensor(s), self.nS)
        return s
