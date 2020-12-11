

### Vanilla Policy Gradient
Monte-Carlo policy gradient relies on an estimated return by Monte-Carlo methods using episode samples to update the policy parameter θ. REINFORCE works because the expectation of the sample gradient is equal to the actual gradient:

∇θJ(θ)=Eπ[Qπ(s,a)∇θlnπθ(a|s)]=Eπ[Gt∇θlnπθ(At|St)]; Because Qπ(St,At)=Eπ[Gt|St,At]
Therefore we are able to measure Gt from real sample trajectories and use that to update our policy gradient. It relies on a full trajectory and that’s why it is a Monte-Carlo method.

The process is pretty straightforward:

1. Initialize the policy parameter θ at random.
2. Generate one trajectory on policy πθ: S1,A1,R2,S2,A2,…,ST.
3. For t=1, 2, … , T:
- Estimate the the return Gt;
- Update policy parameters: θ←θ+αγtGt∇θlnπθ(At|St)


 
