"""
This module has been adapted from and relies on the BuTools project: https://github.com/ghorvath78/butools
"""
import numpy as np
import numpy.matlib as ml
from numpy.random import rand
from butools.map import *

def SamplesFromMAP (D0, D1, k, initial=None, prec=1e-14):
    """
    Generates random samples from a marked Markovian 
    arrival process.
    
    Parameters
    ----------
    D0,D1 : matrices of shape(M,M) of the MAP
    K : integer
        The number of samples to generate.
    initial: optional, initial state
    prec : double, optional
        Numerical precision to check if the input MMAP is
        valid. The default value is 1e-14.

    
    Returns
    -------
    x : matrix, shape(K,2)
        The random samples. Each row consists of two 
        columns: the inter-arrival time and the type of the
        arrival.        
    """
    D=(D0,D1)

    if not CheckMAPRepresentation (D0,D1):
        raise Exception("SamplesFromMMAP: Input is not a valid MMAP representation!")    

    N = D[0].shape[0]
    
    if initial==None:
        # draw initial state according to the stationary distribution
        stst = CTMCSolve(SumMatrixList(D)).A.flatten()
        cummInitial = np.cumsum(stst)
        r = rand()
        state = 0
        while cummInitial[state]<=r:
            state+=1
    else:
        state = initial

    # auxilary variables
    sojourn = -1.0/np.diag(D[0])
    nextpr = ml.matrix(np.diag(sojourn))*D[0]
    nextpr = nextpr - ml.matrix(np.diag(np.diag(nextpr)))
    for i in range(1,len(D)):
        nextpr = np.hstack((nextpr, np.diag(sojourn)*D[i]))
    nextpr = np.cumsum(nextpr,1)
    
    if len(D)>2:
        x = np.empty((k,2))
    else:
        x = np.empty(k)

    for n in range(k):
        time = 0

        # play state transitions
        while state<N :
            time -= np.log(rand()) * sojourn[state]
            r = rand()
            nstate = 0
            while nextpr[state,nstate]<=r:
                nstate += 1
            state = nstate
        if len(D)>2:
            x[n,0] = time
            x[n,1] = state//N
        else:
            x[n] = time
        state = state % N
    
    return x, state

def MapMean (D0, D1):
    return MarginalMomentsFromMAP(D0,D1,1)[0]


#D0 = ml.matrix([[-0.17, 0, 0, 0.07],[0.01, -0.78, 0.03, 0.08],[0.22, 0.17, -1.1, 0.02],[0.04, 0.12, 0, -0.42]])
#D1 = ml.matrix([[0, 0.06, 0, 0.04],[0.04, 0.19, 0.21, 0.22],[0.22, 0.13, 0.15, 0.19],[0.05, 0, 0.17, 0.04]])
#iat, s = SamplesFromMAP(D0, D1, 1, initial=1)

rate=10
# erlang
def make_erlang2 (rate):
    x=rate*2
    D0 = ml.matrix([[-x, x],[0.0, -x]])
    D1 = ml.matrix([[0, 0],[x, 0]])
    return (D0, D1)

#hyper
def make_hyper (rate):
    l2=0.75*rate
    l1=2*l2
    p=0.5
    q=1.0-p
    D0 = ml.matrix([[-l1, 0],[0, -l2]])
    D1 = ml.matrix([[p*l1,q*l1],[p*l2, q*l2]])
    return (D0, D1)

def make_mmpp2 (rate):
    m1=0.75*rate
    m2=10*rate
    l12=0.1
    l21=3.5
    
    D0 = ml.matrix([[-l12-m1, l12],[l21, -l21-m2]])
    D1 = ml.matrix([[m1, 0],[0, m2]])
    return (D0, D1)

if __name__ == "__main__":
    #Mean rate: 10.0 - SCV: 0.5
    #Mean rate: 10.0 - SCV: 1.2222222222222219
    #Mean rate: 10.069444444444441 - SCV: 1.5877814088598385
    D0,D1 = make_erlang2(rate)
    mean,m2 = MarginalMomentsFromMAP(D0,D1,2)
    var = m2 - mean**2
    scv = var/mean**2
    print(f"Mean rate: {1.0/mean} - SCV: {scv}")

    D0,D1 = make_hyper(rate)
    mean,m2 = MarginalMomentsFromMAP(D0,D1,2)
    var = m2 - mean**2
    scv = var/mean**2
    print(f"Mean rate: {1.0/mean} - SCV: {scv}")

    D0,D1 = make_mmpp2(rate)
    mean,m2 = MarginalMomentsFromMAP(D0,D1,2)
    var = m2 - mean**2
    scv = var/mean**2
    print(f"Mean rate: {1.0/mean} - SCV: {scv}")

