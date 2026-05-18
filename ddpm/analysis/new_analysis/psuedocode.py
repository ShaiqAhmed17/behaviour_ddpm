#psuedocode

'''
X is the teacher dataset of shape [trials, samples, timesteps, neurons]
Y is the student dataset of shape [trials, samples, timesteps, neurons]
    neurons = M-space, i.e. behavioural nullspace
    trials are differentiated based on which input was provided
    samples (trajectories) are randomly generated

X[n] and Y[n] can both be seen as samples from a probability flow, in response to input[n]
    Size [samples, timesteps, neurons]

So, for a sample *indices* k1 and k2,
    X[n,k1] and Y[n,k2] are sized [timesteps, neurons] -- these are individual trajectories in neural state space
    Note: k is not a global indexer - there's no ordering across them

We want to find a transform matrix that
    - Minimises average distance between these sets of trajectories
    - For the most optimistic pairing of k1 and k2


'''

