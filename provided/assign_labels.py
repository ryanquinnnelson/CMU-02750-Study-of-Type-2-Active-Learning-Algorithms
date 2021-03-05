import numpy as np 

def assign_labels(L,u,v,T,n_samples):
    """Assign labels to every leaf according to the label of the subtree's root node

    :param L: array of predicted labels for each node
    :param u: current node
    :param v: subtree's root node
    :param T: tree- 3 element list, see dh.py for description
    :param nsample: number of samples, 1000

    :returns L: array of predicted label for each node
    """

    link = T[0]
    if u < n_samples:
        L[u] = L[v]
        return L
    else:
        left = link[u-n_samples,0]
        L = assign_labels(L,left,v,T,n_samples)
        right = link[u-n_samples,1]
        L = assign_labels(L,right,v,T,n_samples)
        return L 