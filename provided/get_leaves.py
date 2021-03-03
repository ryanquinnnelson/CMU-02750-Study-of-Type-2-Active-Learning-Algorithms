import numpy as np 

def get_leaves(leaves,v,T,n_samples):
    """Get all leaf nodes in the subtree Tv rooted at v

    :param leaves: previously found leaves
    :param v: current root node
    :param T: Tree- 3 element list, see dh.py for description
    :param n_samples: number of samples

    :returns leaves: leaves in the subtree Tv rooted at v"""

    link = T[0]
    if v < n_samples:
        leaves.append(v)
        return leaves
    else:
        left = link[v-n_samples,0]
        leaves = get_leaves(leaves,left,T,n_samples)
        right = link[v-n_samples,1]
        leaves = get_leaves(leaves,right,T,n_samples)
        return leaves
