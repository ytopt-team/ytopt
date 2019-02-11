import numpy as np
import numpy.ma as ma
import math
import random
import sys

np.random.seed(0)
random.seed(0)

"""
Notations:

We consider m resources spread across a maximum of n storage nodes.
L = [ l1, ... lm ] is a row vector representing the load of each resource.
S = [ s1, ... sm ] is a row vector representing the size of each resource.

We consider that L is normalized (i.e. lk are in [0,1] and their sum is 1),
same for S (sk are in [0,1] and their sum is 1).

A = [ a1, ... an ] is a row vector representing the state of each storage node,
  such that ak = 0 if the storage node k is turned off, ak = 1 otherwise.

    [[ r1,1 ... r1,n ]
R =      |         |
     [ rm,1 ... rm,n ]] is a matrix representing the initial placement of the
  m resources across the n nodes. That is, ri,j = 1 if resource i is located
  on node j, ri,j = 0 otherwise. Note that a resource can only be located on
  one node.
Y = a matrix similar to R representing the new placement proposed by the system.
  This matrix should be either an exact placement (i.e. 0s and 1s like in R) or
  a probabilistic placement, i.e. Y[i,j] is between 0 and 1 and the sum of elements
  in a row equals to 1 (row i becomes a representation of the probability to find
  resource i in node j=1,...,m).
"""

def __generate_test_config(m,n):
    """
    This function generates a dataset for testing.
    """
    L = np.random.rand(m)
    L /= np.sum(L)
    S = np.random.rand(m)
    S /= np.sum(S)
    A0 = np.random.randint(2, size=n)
    if(np.sum(A0) == 0):
        A0[0] = 1
    max_n0 = np.sum(A0)
    A1 = np.random.randint(2, size=n)
    if(np.sum(A1) == 0):
        A1[0] = 1
    max_n1 = np.sum(A1)
    R = np.zeros((m,n))
    for i in range(0,m):
        k = random.randint(0,max_n0-1)
        for j in range(0,n):
            if A0[j] == 1:
                if k == 0:
                    R[i,j] = 1.0
                k -= 1
    Y = np.random.rand(m,n)
    for j in range(0,n):
        Y[:,j] /= np.sum(Y[:,j])
    return (L, S, A1, R, Y)


def load_imbalance(Y, L, A):
    """
    This function computes the load imbalance of the proposed configuration.
    It computes the variance of load across active nodes. This variance is
    minimized to 0 when all the active nodes have the same load. Because the
    loads sum to 1, the load in each storage node is less than 1, therefore
    the variance will be between 0 and 0.25 (Popoviciu's inequality on variance),
    so we multiple it by 4 to get something between 0 and 1.
    """
    m, n = Y.shape
    YA = np.argmax(Y*A, axis=1)
    Loads = np.zeros((n))
    for i in range(0,m):
        Loads[YA[i]] += L[i]
    ActiveLoads = ma.masked_array(Loads, mask=(1-A))
    return 4*np.var(ActiveLoads)

def __test_load_imbalance(L, S, A, R, Y):
    """
    This function tests the above load_imbalance function.
    """
    return load_imbalance(Y,L,A)

def constraints(Y, A):
    """
    This function computes how well the proposed configuration satisfies the
    constraints that no resources should be located on nodes that are leaving.
    To do so we compute the proportion of misplaced resources in Y (resources
    that have been placed on a non-active node) and divide by m to normalize
    between 0 and 1.
    """
    m,n = Y.shape
    omA = 1 - A
    mult = np.dot(Y, np.transpose(np.expand_dims(omA,0))) 
    C = np.sum(mult)/m
    return C

def __test_constraints(L, S, A, R, Y):
    """
    This function tests the above constraints function.
    """
    return constraints(Y,A)

def transfers(Y, R, S):
    """
    This function computes the cost of data transfers required to move the data
    from configuration R to configuration Y. To do that we find out which resources
    moved between one configuration to the other, we multiply by the size to get the
    proportion of data that moved.
    """
    m, n = R.shape
    RYS = (R-Y) * np.expand_dims(S,1)
    T = np.sum(np.abs(RYS))/m
    return T

def __test_transfers(L, S, A, R, Y):
    """
    This function tests the above transfers function.
    """
    return transfers(Y,R,S)

def choice(Y):
    """
    This function penalizes not making a concrete choice of a node
    on which to place a resource.
    """
    m,n = Y.shape
    max_entropy = - m*math.log(1.0/n)
    log_Y = np.ma.log(Y)
    return - np.sum(Y * log_Y) / max_entropy

def __test_choice(L, S, A, R, Y):
    """
    This function tests the above choice function.
    """
    return choice(Y)

def loss(L, S, A, R, Y, alpha, beta, gamma, delta):
    """
    This function computes the final loss using the above functions.
    """
    return alpha*load_imbalance(Y,L,A) + beta*constraints(Y,A) + gamma*transfers(Y,R,S) + delta*choice(Y)

def __test_loss(L, S, A, R, Y):
    return loss(L, S, A, R, Y, 0.25, 0.25, 0.25, 0.25)



if __name__ == '__main__':
    m, n = 32, 8
    L,S,A,R,Y = __generate_test_config(m,n)
    l = __test_load_imbalance(L,S,A,R,Y)
    print("test_load_imbalance\t| loss = "+str(l))
    l = __test_constraints(L,S,A,R,Y)
    print("test_constraints\t| loss = "+str(l))
    l = __test_transfers(L,S,A,R,Y)
    print("test_transfers\t\t| loss = "+str(l))
    l = __test_choice(L,S,A,R,Y)
    print("test_choice\t\t| loss = "+str(l))
    l = __test_loss(L,S,A,R,Y)
    print("test_loss\t\t| loss = "+str(l))
