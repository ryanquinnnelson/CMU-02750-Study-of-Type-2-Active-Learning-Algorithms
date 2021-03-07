import numpy as np
import packages.dh.helper as helper
import random


# tested
def _proportional_selection(P, T, num_samples):
    """
    Select a node from the pruning, proportional to the size of subtree rooted at each node.
    :param P:
    :param T:
    :param num_samples:
    :return:
    """

    # for each subtree in P, get the number of leaf nodes in that subtree
    num_leaves = np.zeros(len(P))
    for i in range(len(P)):
        leaves = helper.get_leaves([], P[i], T, num_samples)
        num_leaves[i] = (len(leaves))

    # set weights for each node in P, proportional to number of leaves under that node / number of nodes in T
    p = num_leaves / num_samples
    selected = np.random.choice(P, 1, p=p)
    return selected[0]


def _confidence_adjusted_selection(T, num_samples):
    """
    Select a node from P biasing towards choosing nodes in areas where the observed labels are less pure.

    :param T:
    :param num_samples:
    :return:
    """

    pass


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

    # perform i iterations
    for j in range(budget):

        # step 1
        selected_P = set()  # ?? should be set instead of list
        for b in range(batch_size):
            # TODO: select a node from P proportional to the size of subtree rooted at each node (DONE)
            v = _proportional_selection(P, T, num_samples)
            selected_P.add(v)

            # TODO: pick a random leaf node from subtree Tv and query its label (DONE)
            v_leaves = helper.get_leaves([], v, T, num_samples)
            z = random.choice(v_leaves)
            label_z = y_true[z]

            # TODO: update empirical counts and probabilities for all nodes u on path from z to v (DONE)
            n, pHat1 = helper.update_empirical(n, pHat1, v, z, label_z, T)
            # print('pHat1',pHat1)
            # print('n',n)

        # step 2
        for p in selected_P:
            # TODO: update admissible A and compute scores; find best pruning and labeling (DONE)
            P_best, L_best = helper.best_pruning_and_labeling(n, pHat1, p, T, num_samples)

            # TODO: update pruning P and labeling L
            # update pruning
            P_without_s = P[P != p]  # remove p from P using a mask
            P = np.union1d(P_without_s, P_best)

            # assign label L_best to all u in P_best
            for u in P_best:
                L[u] = L_best

        # TODO: temporarily assign labels to every leaf and compute error (DONE)
        L_temp = L.copy()
        for v in P:
            L_temp = helper.assign_labels(L_temp, v, v, T, num_samples)  # assign each leaf in Tv the label L(v)
        error_i = helper.compute_error(L_temp[:num_samples], y_true)  # compute error of leaf nodes only
        error.append(error_i)

    # after all iterations
    # assign final labeling based on best pruning
    for j in range(len(P)):
        L = helper.assign_labels(L, P[j], P[j], T, num_samples)  # assign each leaf in Ti the label L(i)
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
