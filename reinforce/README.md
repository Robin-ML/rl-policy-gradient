

### REINFORCE 
REINFORCE relies on an estimated return by ```Monte-Carlo methods``` using episode samples to update the policy parameter ```θ```. REINFORCE works because the expectation of the sample gradient is equal to the actual gradient.

Therefore we are able to measure reward from real sample trajectories and use that to update our policy gradient. It relies on a full trajectory and that’s why it is a Monte-Carlo method.

The process is pretty straightforward:

1. Initialize the policy parameter ```θ``` at random.
2. Generate one trajectory on policy π <sub>θ</sub> : S<sub>1</sub>,A<sub>1</sub>,R<sub>2</sub>,S<sub>2</sub>,A<sub>2</sub>,…,S<sub>T</sub>.
3. For t=1, 2, … , T:
- Estimate the the return G<sub>t;
- Update policy parameters: θ ← θ + αγ<sup>t</sup>G<sub>t</sub>∇<sub>θ</sub>ln π<sub>θ</sub>(A<sub>t</sub>|S<sub>t</sub>)


 
