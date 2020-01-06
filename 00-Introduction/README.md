# Introduction

## Summary

- RL is different from (un)supervised learning:
      - only a reward signal
      - delayed feedback
      - time matters: sequential, non-iid data
      - agent actions affect subsequent data it receives

- RL **reward hypothesis**: all goes can be described by the
  maximization of expected cumulative reward.
  
  *(Question: in practice, seems hard to design rewards where there is one
  final goal whose total reward can always be surpassed by taking
  infinitely many indirectly relevant actions with small rewards)*

- Information / Markov state - contains all useful info from history.
  $$P(S_{t+1} | S_t) = P(S_{t+1} | S_1, ..., S_t)$$

- State of environment $S^e_t$ is Markov by definition.  State of
  agent $S^a_t$ is not necessarily Markov / can be imperfect.
  Example: helicopter only has positional info, no velocity.

  *(Question: in practice, how much work do we have to put into trying
  to get a Markovian agent state representation / does it affect
  performance significantly?)*
  
- Fully observable environment (formally, a Markov Decision Process -
  MDP): 
  Observation = state of agent = state of environment
  $$O_t = S^a_t = S^e_t$$
  
- Partially observable environment (POMDP): agent must construct its
  own env, e.g. full history, probabilistic/Bayesian rep of states,
  recurrent neural network
  $$S^a_t \neq S^e_t$$
  
- Main components of RL agent:
      - **policy** - the behavior, how agent chooses action.  Maps state to action.
      - **value function** - how good is each state and/or action. Predict expected future reward.
      - **model** - (optional) agent's representation of environment.  Has two parts:
          - transitions: predicts next state (dynamics)
          - rewards: predicts next immediate reward

- Categories of RL agents:
      - Value based: no policy (implicit), value function ("stores the value function")
      - Policy based: policy, no value function ("stores the policy")
      - Actor Critic: policy + value function ("stores both")
      - Model free vs. Model based
  
- Learning and Planning: in RL, env is initially unknown and agent has
  to interact with env to improve its policy.  In planning, model of
  env *is* known, e.g. rules of game, and agent can perform
  computations with model to decide on actions

- Prediction vs Control:
      - prediction: evaluate future given policy
      - control: optimize future (find best policy, first need to solve
        prediction problem)
 
## Questions



## Resources

- David Silver's RL Course Lecture 1 - Introduction to Reinforcement
  Learning ([video](https://www.youtube.com/watch?v=2pWv7GOvuf0),
  [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf))
