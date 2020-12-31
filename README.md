![license MIT](https://img.shields.io/badge/licence-MIT-green)


![Reinforcement Learning](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/1024px-Reinforcement_learning_diagram.svg.png)

## Policy Gradient Methods
We focuses on a particular family of reinforcement learning algorithms that use policy gradient methods. They are designed to be easily adaptable for reinforcement learning environments (like [gym](https://github.com/openai/gym)). 

The goal of reinforcement learning is to find an ```optimal behavior strategy ```for the agent to obtain optimal rewards. The policy gradient methods target at modeling and optimizing the policy directly. The policy is usually modeled with a parameterized function (```θ```), i.e π <sub>θ</sub> (a|s). 
The value of the reward (objective) function depends on this policy and then various algorithms can be applied to optimize ```θ``` for the best reward.

Finding the ```θ``` that maximises the reward is an ```optimisation problem ```. 
Some approaches include:
- Gradient-based:
  - Gradient descent 
  - Conjugate gradient
  - Quasi-newton

- Genetic algorithms
- Hill climbing
- Simplex / amoeba / Nelder Mead


### Tensorflow versions
The master branch supports Tensorflow 2 versions of the baseline algorithm A2C/A3C 

### PG Algorithms
* [x] A2C/A3C
* [ ] ACER
* [ ] ACKTR
* [x] GAE
* [x] PPO
* [x] REINFORCE
* [x] TRPO
