

![Poicy Gradient](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/1024px-Reinforcement_learning_diagram.svg.png)

## Policy Gradient Algorithm
We focuses on a particular family of reinforcement learning algorithms that use policy gradient methods.

The goal of reinforcement learning is to find an optimal behavior strategy for the agent to obtain optimal rewards. The policy gradient methods target at modeling and optimizing the policy directly. The policy is usually modeled with a parameterized function respect to θ, π θ(a|s) and the value of the reward (objective) function depends on this policy and then various algorithms can be applied to optimize θ for the best reward.

Finding the θ that maximises the reward is an optimisation problem . 
Some approaches include:
- Gradient-based:
  - Gradient descent 
  - Conjugate gradient
  - Quasi-newton

- Genetic algorithms
- Hill climbing
- Simplex / amoeba / Nelder Mead


### Tensorflow versions
The master branch supports Tensorflow from version 1.4 to 1.14. For Tensorflow 2.0 support, please use tf2 branch.

### PG Algorithms
- A2C/A3C
- ACER
- ACKTR
- DDPG
- Natural PG
- PPO
- TRPO
