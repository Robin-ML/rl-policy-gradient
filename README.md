
## Policy Gradient Algorithm
The goal of reinforcement learning is to find an optimal behavior strategy for the agent to obtain optimal rewards. The policy gradient methods target at modeling and optimizing the policy directly. The policy is usually modeled with a parameterized function respect to θ, πθ(a|s). The value of the reward (objective) function depends on this policy and then various algorithms can be applied to optimize θ for the best reward.

The reward function is defined as:

<img src="https://render.githubusercontent.com/render/math?math=j()=^{i +\pi} =x+1">

where dπ(s) is the stationary distribution of Markov chain for πθ (on-policy state distribution under π). For simplicity, the parameter θ would be omitted for the policy πθ when the policy is present in the subscript of other functions; for example, dπ and Qπ should be dπθ and Qπθ if written in full.



### Tensorflow versions
The master branch supports Tensorflow from version 1.4 to 1.14. For Tensorflow 2.0 support, please use tf2 branch.

### PG Algorithms
- A2C/A3C
- ACER
- ACKTR
- DDPG
- Vanilla PG
- PPO
