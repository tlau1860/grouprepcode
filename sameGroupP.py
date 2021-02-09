
def sameGroupP(n, prob, part):
    """
    Calculates marginal posterior probabilities of relevant partitions for each possible agent pairing

    INPUTS
    n - number of agents
    prob - pz output from lgmDiscrete
    part - Z output from lgmDiscrete

    OUTPUTS
    A - upper triangle of an n x n number of agents matrix where [i,j] is the probability that agents i and j are in the same latent group

    """

    A = np.zeros([n,n])
    for a in range(n):
        for b in range(n):
            if a == b:
                A[a, b] = 1
            elif a<b:
                for i in range(len(part)):
                    for j in range(len(part[i])):
                        if (a+1) in part[i][j] and (b+1) in part[i][j]:
                            A[a, b] += prob[i]

    return A
