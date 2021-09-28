"""
This is a python adaptation of the latent group model.
It is 1-indexed, so C needs to be 1-indexed (e.g., if v is 2, C should have 1's and 2's).
Uncomment the part at the bottom and run to see an example.

USAGE:
    [P, pz, Z] = lgmDiscrete(C, v, alpha)

INPUTS:
    C - An np.array of discrete choices on N trials made by M rows of agents
        Example: C = np.array([[2,2,1], [1,1,2], [2,1,2], [2,1,2]])
    v - number of options on each trial
        Example: v = 2
    alpha - concentration parameter as an np.array. Defaults to np.linspace(1e-5,10,6)
        Example: alpha = np.array([2])

OUTPUTS:
    P - [M x N x v] array of choice probabilities for each agent on each trial
    pz - Probability of each partition
    Z - Corresponding partitions to pz

tlau, Feb 2021
"""

import numpy as np
import scipy.special
import scipy.misc
from pprint import pprint


def partition(vec):
    if len(vec) == 1:
        yield [ vec ]
        return
    first = vec[0]
    for smaller in partition(vec[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller

def SetPartition(N):
    vector = list(range(1,N+1))
    Z = []
    for n, p in enumerate(partition(vector), 1):
        m = p
        Z.append(m)
    return Z

def crpLogProp(T, alpha):
    """
    Log probability of a partition under the CRP.
    USAGE: logp = crpLogProb(T,alpha)
    INPUTS:
      T - vector of counts for each group in the partition.
      alpha - concentration parameter
    """

    K = len(T[0])
    N = sum(T[0])
    combinedvector = np.append(T, alpha)
    prob = K * np.log(alpha) + np.sum(scipy.special.gammaln(combinedvector)) - scipy.special.gammaln(sum([N],alpha))
    return prob



def lgmDiscrete(C,v,alpha):

    # Initialize by finding number of agents (M) and number of N choices
    M = C.shape[0]
    N = C.shape[1]

    # Find all possible partitions given number of agents (Bell's number)
    Z = SetPartition(M)

    ## uniform distribution
    g = 1

    # If alpha isn't specified, set it to a range
    if str(alpha) == 'none':
        alpha = np.linspace(1e-5,10,6)
    A = alpha.shape[0]
    # Generate array that is the number of possible partitions by number of
    # agents by number of choices observed by possible choices
    q = np.zeros((v,N,len(Z), M))

    # Construct prior
    logp = np.zeros((len(Z), A))

    for j in range(len(Z)):
        h = Z[j]
        K = len(h)

        # Compute the prior
        z = np.zeros((M, 1))
        T = np.zeros((1, K))

        ## For the particular partition under consideration (Z[j]),
        ## generate z, a vector of each agent's group membership, and
        ## T, a vector of the number of agents in each group for that partition.
        ## If two possible groups, then T will be length 2
        for k in range(K):
            # for however many agents in the nth partition, designate these agents as being of the same group
            # add 1 because 0-indexed
            for a in range(len(h[k])):
                   z[h[k][a]-1] = k+1
            T[0][k] = len(h[k])

        ## Fill in vector of priors (logp) for each partition across all
        ## alphas (A)
        for a in range(A):
            logp[j,a] = logp[j,a] + crpLogProp(T, alpha[a])

        # Compute likelihood
        L = np.zeros((N,K,v))
        ## For each set of choices observed
        for n in range(N):
            ## generate whether or not there are missing values
            ix = ~np.isnan(C[:,n])
            ## for the number of partitions
            for k in range(K):
                ## Find the agents of that particular partition whose choices we observe
                f = (z[ix]==k+1)
                ## update the partition probability by incorporating the number of agents of that position, sum(f)
                logp[j,:] = logp[j,:] + scipy.special.gammaln(v*g) - scipy.special.gammaln(sum(f)+g*v)
                ## for the particular choice in the possible choice set
                for c in range(v):
                    ## of the agents in the same partition, which agents agree with each other?
                    wet = np.array([C[ix,n]==c+1])
                    ## update count of likelihood
                    L[n,k,c] = sum(f * wet.T)
                    ## update logp
                    logp[j,:] = logp[j,:] + scipy.special.gammaln(g + L[n,k,c]) - scipy.special.gammaln(g)

        # Predictive probability
        for n in range(N):
            for m in range(M):
                q[:,n,j,m] = (g + L[n,int(z[m])-1,:])/(g*v + sum(L[n,int(z[m])-1,:]))

    # Normalize prior and marginalize over alpha
    pz = np.exp(logp - scipy.misc.logsumexp(logp[:]))
    pz = np.sum(pz, axis=1)


    # choice probabilities for missing trials
    if q.size != 0:
        P = np.zeros((v,M,N))
        for j in range(len(Z)):
            Q = np.squeeze(q[:,:,j,:])
            # Transpose because python does squeezing differently
            Q = np.transpose(Q,(0,2,1))
            P = P + pz[j]*Q

    return P, pz, Z


## Uncomment below and run to see example.
# C = np.array([[2,2,1], [1,1,2], [2,1,2], [2,1,2]])
# v = 2
# alpha = np.array([2])
# [P, pz, Z] = lgmDiscrete(C, v, alpha)
# print("Choice behavior")
# print(C)
# print ('\n'"Choice probabilities")
# print(P)
# print ('\n'"posterior probability of each partition")
# print (pz)
# print ('\n'"Partitions")
# pprint(Z)
