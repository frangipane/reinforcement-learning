"""
Delayed effect environment -- Hindsight Credit Assignment Fig 2 (center)
"""
import numpy as np
import torch
import gym
from gym.envs.toy_text.discrete import DiscreteEnv


class DelayedEffectEnv(DiscreteEnv):
    """
    State state presents a choice of two actions, followed by an aliased
    chain (a POMPDP), with the consequence of the initial choice apparent 
    only in the final state.

    One of the final states is rewarding (r=1), the other penalizing (r=-1).
    The middle states contain white noise rewards of standard deviation sigma.

    First observation: 0
    Aliased states: 1...N
    Two possible final observations: N+1 (rewarding), N+2 (penalizing).

    Each state has 2 allowed actions, but state 0 is the only state for
    which an action has non-trivial consequences on the final state:
    if a_0 == 0 ==> final state = N+1 (rewarding)
    if a_0 == 1 ==> final state = N+2 (penalizing)

    Transition probabilities:
    
    ** For any state s_{i}, excluding states N..N+2:
    P(s_{i+1} | s_{i}, a=0) = 1
    P(s_{i+1} | s_{i}, a=1) = 1

    ** from state N (a subscript denotes timestep):
    P(s_{N+1} | s_N, a_N={0,1}, a_0=0) = 1.0
    P(s_{N+2} | s_N, a_N={0,1}, a_0=1) = 1.0

    ** final states are absorbing
    P(s_{N+1} | s_{N+1}, a={0,1)) = 1.0
    P(s_{N+2} | s_{N+2}, a={0,1)) = 1.0

    Args
    ----
    n : int
        number of states in aliased chain.  Total number of states is N+3.
        n > 0.

    sigma : float
        standard deviation of white noise rewards in aliased chain

    final_reward : float > 0
        magnitude of reward for reaching final states.  Rewarding final state
        has a reward = final_reward, and penalizing final state has 
        reward = -1*final_reward.

    OHE_obs : bool
        if False, obs is an integer >= 0, the default from the discrete space. 
        if True, convert observation to a one hot encoded torch tensor of length n,
        the number of states in the MDP
    """
    def __init__(self, n=3, sigma=0.0, final_reward=1.0, OHE_obs=True):
        if n < 1:
            raise ValueError("n, the number of aliased states, must be greater than 0")

        nS = n+3
        nA = 2
        self.rewarding_final_state = n+1
        self.penalizing_final_state = n+2
        self.nS = nS
        self.sigma = 0.0
        self.final_reward = 1.0
        self.first_action = None
        self._OHE_obs = OHE_obs

        # always start at first state
        isd = np.zeros(nS)
        isd[0] = 1.0

        # P is a dict of dict of lists where
        # P[s][a] == [(probability, nextstate, done),...]
        P = {o: {} for o in range(nS)}

        for o in range(n):
            P[o][0] = [(1.0, o+1, False)]
            P[o][1] = [(1.0, o+1, False)]

        # penultimate state (a hack for compatibility with the step API.
        # first action determines what n transitions to (and the reward).
        P[n][0] = None
        P[n][1] = None

        # final state
        P[self.rewarding_final_state][0] = [(1.0, self.rewarding_final_state, True)]
        P[self.rewarding_final_state][1] = [(1.0, self.rewarding_final_state, True)]
        P[self.penalizing_final_state][0] = [(1.0, self.penalizing_final_state, True)]
        P[self.penalizing_final_state][1] = [(1.0, self.penalizing_final_state, True)]

        super().__init__(nS, nA, P, isd)

    def step(self, a):
        assert self.action_space.contains(a)        
        transitions = self.P[self.s][int(a)]

        if self.first_action is None:
            self.first_action = a

        if self.s == self.nS - 3:  # the last aliased state
            if self.first_action == 0:
                o = self.rewarding_final_state
                r = self.final_reward
            else:
                o = self.penalizing_final_state
                r = -1 * self.final_reward
            d = True
            p = 1.0
        else:
            p, o, d = transitions[0]  # there is only a single transition
            r = np.random.normal(0, self.sigma)
        self.s = o

        if self._OHE_obs:
            o = torch.nn.functional.one_hot(torch.as_tensor(o), self.nS)
        return (o, r, d, {"prob": p, "first_action": self.first_action})

    def reset(self):
        o = super().reset()
        if self._OHE_obs:
            o = torch.nn.functional.one_hot(torch.as_tensor(o), self.nS)
        return o
