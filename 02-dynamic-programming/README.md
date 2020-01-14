# Dynamic Programming (DP)

## Summary

Planning problem: you're given the full dynamics (transition
probabilities) of the MDP.

Dynamic programming solves 2 situations that arise in the planning problem:
(1) prediction / policy evaluation
    input: full MDP
    output: evaluate the value function for that policy
(2) control
    input: full MDP
    output: optimal value function and policy


## Policy evaluation

Given MDP and policy, evaluate rewards

Solutionn: iterate over value functions using Bellman expectation eqn
until convergence to true value fxn $v_{\pi}$.

Method: One step lookahead -- evaluate v(s) based on the lookahead
values of its neighbors' values and the one step reward.

Note, in the process of evaluating a policy, the value function can
help us figure out a better policy.

## Policy iteration

Given a policy $\pi$, cycle through 2 steps:
1. Evaluate the policy: estimate $v_{\pi}(s)$
2. Improve the policy by acting greedily w.r.t. $v_{\pi}$. Generate $\pi’ >
   \pi$.

Iterate over steps 1-2 -- guaranteed convergence to optimal policy $\pi_*$.

Notes on convergence: Silver shows acting greedily will always result
in new value >= previous values (acting greedily improves on previous
policy, won’t result in worse performance).  If improvements stop,
then the (one stpe lookahead form of) Bellman optimality eqn has been
satisfied $v_{\pi}(s) = max_a q_{\pi}(s,a).

## Value iteration


## Questions



## Lectures and Readings

- David Silver's RL Course Lecture 3 - Planning by dynamic programming
  ([video](https://www.youtube.com/watch?v=Nd1-UUMVfz4),
  [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf))

- Sutton and Barto -
  [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) -
  - Chapter 4: Dynamic Programming

- Berkeley Deep RL bootcamp - Core Lecture 1 Intro to MDPs and Exact
  Solution Methods -- Pieter Abbeel
  ([video](https://www.youtube.com/watch?v=qaMdN6LS9rA) |
  [slides](https://drive.google.com/open?id=0BxXI_RttTZAhVXBlMUVkQ1BVVDQ))


## Exercises
