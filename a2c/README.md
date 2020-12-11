### A3C
Asynchronous Advantage Actor-Critic (Mnih et al., 2016), short for A3C, is a classic policy gradient method with a special focus on parallel training.

In A3C, the critics learn the value function while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is designed to work well for parallel training.

Let’s use the state-value function as an example. The loss function for state value is to minimize the mean squared error, Jv(w)=(Gt−Vw(s))2 and gradient descent can be applied to find the optimal w. This state-value function is used as the baseline in the policy gradient update.

Here is the algorithm outline:

1. We have global parameters, θ and w; similar thread-specific parameters, θ′ and w′.
2. Initialize the time step t=1
3. While T≤TMAX:
- Reset gradient: dθ=0 and dw=0.
- Synchronize thread-specific parameters with global ones: θ′=θ and w′=w.
- tstart = t and sample a starting state st.
- While (st != TERMINAL) and t−tstart≤tmax:
   - Pick the action At∼πθ′(At|St) and receive a new reward Rt and a new state st+1.
   - Update t=t+1 and T=T+1
- Initialize the variable that holds the return estimation R={0Vw′(st)if st is TERMINALotherwise
- For i=t−1,…,tstart:
   - R←γR+Ri; here R is a MC measure of Gi.
   - Accumulate gradients w.r.t. θ′: dθ←dθ+∇θ′logπθ′(ai|si)(R−Vw′(si));
- Accumulate gradients w.r.t. w’: dw←dw+2(R−Vw′(si))∇w′(R−Vw′(si)).
- Update asynchronously θ using dθ, and w using dw.

A3C enables the parallelism in multiple agent training. The gradient accumulation step (6.2) can be considered as a parallelized reformation of minibatch-based stochastic gradient update: the values of w or θ get corrected by a little bit in the direction of each training thread independently.

### A2C
[paper|code]

A2C is a synchronous, deterministic version of A3C; that’s why it is named as “A2C” with the first “A” (“asynchronous”) removed. In A3C each agent talks to the global parameters independently, so it is possible sometimes the thread-specific agents would be playing with policies of different versions and therefore the aggregated update would not be optimal. To resolve the inconsistency, a coordinator in A2C waits for all the parallel actors to finish their work before updating the global parameters and then in the next iteration parallel actors starts from the same policy. The synchronized gradient update keeps the training more cohesive and potentially to make convergence faster.