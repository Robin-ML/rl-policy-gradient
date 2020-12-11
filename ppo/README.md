
### PPO 
proximal policy optimization (PPO) simplifies it by using a clipped surrogate objective while retaining similar performance.

First, let’s denote the probability ratio between old and new policies as:

r(θ)=πθ(a|s)πθold(a|s)

Then, the objective function of TRPO (on policy) becomes:

JTRPO(θ)=E[r(θ)A^θold(s,a)]
Without a limitation on the distance between θold and θ, to maximize JTRPO(θ) would lead to instability with extremely large parameter updates and big policy ratios. PPO imposes the constraint by forcing r(θ) to stay within a small interval around 1, precisely [1−ϵ,1+ϵ], where ϵ is a hyperparameter.

JCLIP(θ)=E[min(r(θ)A^θold(s,a),clip(r(θ),1−ϵ,1+ϵ)A^θold(s,a))]
The function clip(r(θ),1−ϵ,1+ϵ) clips the ratio to be no more than 1+ϵ and no less than 1−ϵ. The objective function of PPO takes the minimum one between the original value and the clipped version and therefore we lose the motivation for increasing the policy update to extremes for better rewards.