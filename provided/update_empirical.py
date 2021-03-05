import numpy as np 

def update_empirical(n,p1,v,z,l,T):
    ''' Update empirical counts and probabilities for all nodes u on path

    :param n: array with number of points sampled in the subtree rooted at each node
    :param p1: array with fraction of label = 1 in the subtree rooted at each node
    :param v: selected subtree Tv's root node
    :param z: selected leaf node in subtree Tv
    :param l: queried label for node z
    :param T: tree- 3 element list, see dh.py for details

    :returns n: array with updated number of points sampled in the subtree rooted at each node
    :returns p1: array with updated fraction of label = 1 in the subtree rooted at each node
    '''

    parents = T[2]
    while z <= v and z != 0:
        l1 = n[z] * p1[z]
        n[z] += 1 
        p1[z] = (l1+l)/n[z]
        z = np.array([parents[int(z)]])
    return n, p1

