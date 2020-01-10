# Finite Markov Decision Processes (MDPs)

## Summary

### Intro
- RL problem: sequential decision-making: agent with a goal interacts
  with environment.  How does agent decide what actions to take as it
  experiences the environment?

- We can think about the sequences of interactions as a sequence of
  states $S$, actions $A$ by the agent, and rewards $R$ (affected by
  the agent's actions): $S_0, A_0, R_1, S_1, A_1, ...$.  The rewards
  are defined with respect to the agent's goal.

- The states obey the Markov property: future state is independent of
  past states given present state: $P_{ss’} =
  P[S_{t+1} = s’ | S_t = s]$
  
- Q: Where to draw line b/w agent and environment?  A: it depends, but
  anything that the agent cannot change arbitrarily is part of the
  environment.

- Reward hypothesis: "all goals can be described by the maximiation of
  expected cumulative reward".  Formally, the cumulative reward is
  called the "return", the total discounted reward from time step t:
  $$ G_t = \sum_k=0^\inf \gamma^k R_{t+k+1} $$
  - Reasons to use discount:
    - mathematically convenient: avoids infinite returns cyclic Markov
      processes (but not necessary for episodic MDPs, i.e. with an
      absorbing state, e.g. finishing a maze, game, etc. -- but equiv
      to setting to $\gamma$ 0)
    - we don't have a perfect model of the env, so build in
      uncertainty in future

?? question: in practice, how do we choose gamma?  Why should we do
this?  How sensitive is performance to it?  Do we use it as a way to
indirectly tune behavior/policy?

### Value function characterizes the "goodness" of a state
- formally, the expected return of state $S$: 
  $$v(s) = Expected_value(G_t | S_t = s)$$
- note: Silver covers this in his lecture, but S & B skip MRPs
  straight to MDPs. (In MRP, just talking about measuring v(s), but
  want to maximize v(s) in MDP).

?? question: is $v(s)$ monotonic in $\gamma$ per state?

### Bellman equation in MRPs:

Fundamental property of value functions in RL: satisfy a recursive
decomposition.

Bellman eqn decomposes $v(s)$ into 2 parts: “Immediate reward
$R_{t+1}$ + discounted value of successor state $\gamma v(S_{t+1})$”
$$ v(s) = \mathbb{e}( R_{t+1} + \gamma v(S_{t+1} | S_t = s)$$

#### Solving Bellman eqn
- can express as linear matrix equation (directly solvable for small
number of states, solve iteratively for large number, e.g. with DP,
monte carlo, TD).
- result is a set of (self-consistent) value function for all states!

#### Understanding Bellman eqn
- Can think of as average over all possible values of neighbor states,
  weighted by their probability (plus the expected reward from state
  s->s').

### MDP

Is an MRP with decisions.  Add in finite set of actions $A$.
Trajectory is still an MRP, but now dynamics described by the policy.

#### Policy

What action will an agent take in a state?  Policy maps state to
probability of selecting each action: $\pi(a|s)$.

**state-value function for MDP**

expected return in that state *under a policy $\pi$*

$$
v_{\pi}(s) = \mathbb{e}_{\pi}( G_t | S_t = s )
$$

**action-value function**

expected return in that state under policy $\pi$ and action $a$.  This
is what agent will use to decide what action to take (to optimize the
MDP)!

$$ 
q_{\pi}(s, a) = \mathbb{e}_{\pi}( G_t | S_t = s, A_t = a )
$$

v, q for MDP similarly can be expressed as Bellman equations.


#### Optimal value functions

Optimal value functions tell you what the max reward is that you can
extract from the system.

Optimal state-value function: $v_*(s) = max_{\pi} v_{\pi}(s)$
Optimal action-value function{ $q_*(s, a) = max_{\pi} q_(s, a)$

Once you know the optimal value functions, it's easy to figure out the
optimal policy.


#### Optimal policy

Definition of "optimal policy" \pi_*:

$$
\pi >= \pi’ if v_\pi(s) >= v_\pi’(s), for all s
$$

?? question: why use this definition of optimal policy where value
functions have to be greater in a state by state comparison, rather
than, e.g. the sum of the value functions?

At least one optimal policy exists.

The optimal policy deterministically chooses an action $a$ in state
$s$ that maximizes the optimal action-value function (and chooses
other actions with 0 probability).


### Solve for optimal value functions --> Bellman Optimality Equations

Lookahead again for state value function, but instead of taking
average of lookahead q_values, TAKE THE MAX.

$$
v_*(s) = max_a q_*(s, a)
$$

Similarly, for the optimal action value function, $q_*$:

$$
q_*(s, a) = R^a_s + \gamma \sum_{s’ \elem S} P^a_{ss’} v_*(s’)
$$

Have to average over all the things the environment might do to us
(optimal value fxns of states we end up in) after we take our action.


Combine these two pieces to get v_* as a function of itself ( a 2 step
lookahead) => Bellman Optimality Equation for v*:

$$
v_*(s) = max_a R^a_s + \gamma \sum_{s’ \elem S} P^a_{ss’} v_*(s’)
$$

*Notes*:
- Bellman optimality equation is nonlinear because of the $max$.
- no general closed form solution.


## Questions



## Lectures and Readings

- David Silver's RL Course Lecture 2 - Introduction to Reinforcement
  Learning ([video](https://www.youtube.com/watch?v=lfHX2hHRMVQ),
  [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf))

- Sutton and Barto -
  [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) -
  - Chapter 3: Finite Markov Decision Processes


## Exercises
