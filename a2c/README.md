### A3C
Asynchronous Advantage Actor-Critic (Mnih et al., 2016), short for A3C, is a classic policy gradient method with a special focus on cparallel training```.

In ***A3C***, the critics learn the value function while ```multiple actors``` are trained in parallel and get synced with ```global parameters``` from time to time. Hence, A3C is designed to work well for parallel training.

Let’s use the state-value function as an example. The loss function for state value is to minimize the ```mean squared error```,
 J<sub>v</sub>(w)=(G<sub>t</sub>−V<sub>w</sub>(s))<sup>2</sup> and gradient descent can be applied to find the optimal w. This state-value function is used as the baseline in the policy gradient update.

Here is the algorithm outline:

1. We have global parameters, θ and w; similar thread-specific parameters, θ<sup>′</sup> and w<sup>′</sup>.
2. Initialize the time step t=1
3. While T≤TMAX:
- Reset gradient: dθ=0 and dw=0.
- Synchronize thread-specific parameters with global ones: θ′=θ and w′=w.
- tstart = t and sample a starting state st.
- While (s<sub>t</sub> != TERMINAL) and t−t<sub>start</sub>≤t<sub>max</sub>:
   - Pick the action A<sub>t</sub>∼π<sub>θ′</sub>(At|St) and receive a new reward Rt and a new state s<sub>t+1</sub>.
   - Update t=t+1 and T=T+1
- Initialize the variable that holds the return estimation

- For i=t−1,…,t<sub>start</sub>:
   - R←γR+R<sub>i</sub>; here R is a MC measure of G<sub>i</sub>.
   - Accumulate gradients w.r.t. θ′: dθ←dθ+∇<sub>θ′</sub>logπ<sub>θ′</sub>(a<sub>i</sub>|s<sub>i</sub>)(R−V<sub>w′</sub>(s<sub>i</sub>));
- Accumulate gradients w.r.t. w’: dw←dw+2(R−V<sub>w′</sub>(s<sub>i</sub>))∇<sub>w′</sub>(R−V<sub>w′</sub>(s<sub>i</sub>)).
- Update asynchronously θ using dθ, and w using dw.

A3C enables the parallelism in multiple agent training. The gradient accumulation step (6.2) can be considered as a parallelized reformation of minibatch-based stochastic gradient update: the values of w or θ get corrected by a little bit in the direction of each training thread independently.

### A2C

A2C is a synchronous, deterministic version of A3C; that’s why it is named as “A2C” with the first “A” (“asynchronous”) removed. In A3C each agent talks to the global parameters independently, so it is possible sometimes the thread-specific agents would be playing with policies of different versions and therefore the aggregated update would not be optimal. To resolve the inconsistency, a coordinator in A2C waits for all the parallel actors to finish their work before updating the global parameters and then in the next iteration parallel actors starts from the same policy. The synchronized gradient update keeps the training more cohesive and potentially to make convergence faster.