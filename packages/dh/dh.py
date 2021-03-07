import numpy as np
import packages.dh.helper as helper
import random


# ?? num_samples is correct here?
# ?? is the pruning defined by its subtrees
def _select_random_subtree(P, T, num_samples):
    """
    each subtree has a chance of being chosen in proportion to the leaves of that subtree vs. the leaves in the tree
    p_v = l_v / num_samples
    :param P:
    :param T:
    :param num_samples:
    :return:
    """
    # get nodes representing the root of the subtrees in this pruning
    link = T[0]
    subtrees = []
    for v in P:
        left = link[v - num_samples, 0]  # root of left subtree
        right = link[v - num_samples, 1]  # root of right subtree
        subtrees.append(left)
        subtrees.append(right)

    # choose a subtree randomly using weighted proportions
    sizes_of_subtrees = T[1]
    sizes_of_subtrees_in_P = sizes_of_subtrees[subtrees]  # limit only to subtrees in this pruning
    p = sizes_of_subtrees_in_P / num_samples  # weight for each subtree in this pruning
    return np.random.choice(subtrees, 1, p=p)


def select_case_1(X, y_true, T, budget, batch_size):
    """

    :param X: a x b matrix, data samples
    :param y_true: a x 1 vector, true labels for samples
    :param T: Tree data structure
    :param budget: Number of iterations allowed
    :param batch_size: Number of queries per iteration
    :return:
    """

    # define variables
    num_nodes = len(T[1])  # total nodes in T
    num_samples = len(X)  # total samples in data

    # set scaffolds to fill in
    n = np.zeros(num_nodes)  # number of points sampled from each node
    pHat1 = np.zeros(num_nodes)  # empirical label frequency
    L = np.zeros(num_nodes)  # majority label

    # additional
    error = []  # error for each iteration

    # set initial pruning and labeling for root
    root = num_nodes - 1  # index of root
    P = np.array([root])
    L[root] = 1

    for i in range(budget):

        # part 1
        selected_P = []
        for b in range(batch_size):
            # TODO: select a node from P proportional to the size of subtree rooted at each node (DONE)
            v = _select_random_subtree(P, T, num_samples)
            selected_P.append(v)

            # TODO: pick a random leaf node from subtree Tv and query its label (DONE)
            v_leaves = helper.get_leaves([], v, T, num_samples)
            z = random.choice(v_leaves)
            label_z = y_true[z]

            # TODO: update empirical counts and probabilities for all nodes u on path from z to v (DONE)
            n, pHat1 = helper.update_empirical(n, pHat1, v, z, label_z, T)

        # part 2
        for p in selected_P:
            # TODO: update admissible A and compute scores; find best pruning and labeling
            # ?? what should root be? p?
            P_best, L_best = helper.best_pruning_and_labeling(n, pHat1, p, T, num_samples)

            # TODO: update pruning P and labeling L
            # ?? store on its own? what does it mean to update labeling and pruning P?
            P = P_best
            L = helper.assign_labels(L, P[p], P[p], T, num_samples)

        # ?? start from root? what should v be?
        # TODO: temporarily assign labels to every leaf and compute error
        L_temp = L.copy()
        L_i = helper.assign_labels(L_temp, root, root, T, num_samples)
        error_i = helper.compute_error(L_i, y_true)
        error.append(error_i)

    # assign final labeling based on best pruning
    for i in range(len(P)):
        L = helper.assign_labels(L, P[i], P[i], T, num_samples)

    return L, np.array(error)

# def select_case_2(data, labels, T, budget, batch_size):
#     """DH algorithm where we choose P by biasing towards choosing nodes in areas where the observed labels are less pure
#
#     :param data: Data matrix 1200x8
#     :param labels: true labels 284x1
#     :param T: 3 element tree
#         T[0] = linkage matrix from hierarchical clustering.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#                for details. If you are unfamiliar with hierarchical clustering using scipy, the following is another helpful resource (We won't use dendrograms
#                here, but he gives a nice explanation of how to interpret the linkage matrix):
#                https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
#
#         T[1] = An array denoting the size of each subtree rooted at node i, where i indexes the array.
#                ie. The number of all children + grandchildren + ... + the node itself
#
#         T[2] = dict where keys are nodes and values are the node's parent
#     :param budget: Number of iterations to make
#     :param batch_size: Number of queries per iteration"""
#
#     n_nodes = len(T[1])  # total nodes in T
#     n_samples = len(data)  # total samples in data
#     L = np.zeros(n_nodes, dtype=int)  # majority label
#     p1 = np.zeros(n_nodes)  # empirical label frequency
#     n = np.zeros(n_nodes)  # number of points sampled from each node
#     error = []  # np.zeros(n_samples) #error at each round
#     root = n_nodes - 1  # corresponds to index of root
#     P = np.array([root])
#     L[root] = 1
#
#     for i in range(budget):
#         selected_P = []
#         for b in range(batch_size):
#             # TODO: select a node from P biasing towards choosing nodes in areas where the observed labels are less pure
#             raise NotImplementedError
#
#             # TODO: pick a random leaf node from subtree Tv and query its label
#
#             # TODO: update empirical counts and probabilities for all nodes u on path from z to v
#
#         for p in selected_P:
#             # TODO: update admissible A and compute scores; find best pruning and labeling
#             raise NotImplementedError
#             # TODO: update pruning P and labeling L
#
#         # TODO: temporarily assign labels to every leaf and compute error
#         L_temp = L.copy()
#         raise NotImplementedError
#
#     for i in range(len(P)):
#         L = helper.assign_labels(L, P[i], P[i], T, n_samples)
#
#     return L, np.array(error)


# for testing purposes only
# def select_case_1(X, y_true, T, budget, batch_size):
#     return np.array([np.random.randint(0,2) for i in range(len(T[1]))]), [np.random.uniform(0,1) for i in range(budget)]
#
#
# def select_case_2(X, y_true, T, budget, batch_size):
#     return np.array([np.random.randint(0,2) for i in range(len(T[1]))]), [np.random.uniform(0,1) for i in range(budget)]
