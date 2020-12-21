

### Trust region policy optimization (TRPO)
To improve training stability, we should avoid parameter updates that change the policy too much at one step. TRPO(Schulman, et al., 2015) carries out this idea by enforcing a KL divergence constraint on the size of policy update at each iteration.
Trust regions are defined as the region in which the local approximations of the function are accurate. Ok, but what does that mean? In trust regions, we determine the maximum step size and then we find the local maximum of the policy within the region. 

By continuing the same process iteratively, we find the global maximum. We can also expand or shrink the region based on how good the new approximation is. That way we are certain that the new policies can be trustworthy of not leading to dramatically bad policy degradation. We can express mathematically the above constraint using KL divergence( which you can think of as a distance between two probabilities distributions)

 
