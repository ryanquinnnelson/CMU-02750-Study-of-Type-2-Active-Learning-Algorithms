import numpy as np
import packages.dh.helper as helper


def select_case_1(X, labels, T, budget, batch_size):

    n_nodes = len(T[1])  # total nodes in T
    n_samples = len(X)  # total samples in data

    # set scaffold to fill in
    L = np.zeros(n_nodes)  # majority label
    p1 = np.zeros(n_nodes)  # empirical label frequency
    n = np.zeros(n_nodes)  # number of points sampled from each node




    error = []  # np.zeros(n_samples) #error at each round
    root = n_nodes - 1  # corresponds to index of root
    P = np.array([root])
    L[root] = 1

    for i in range(budget):
        selected_P = []
        for b in range(batch_size):
            # TODO: select a node from P proportional to the size of subtree rooted at each node
            raise NotImplementedError

            ##TODO: pick a random leaf node from subtree Tv and query its label

            # TODO: update empirical counts and probabilities for all nodes u on path from z to v

        for p in selected_P:
            # TODO: update admissible A and compute scores; find best pruning and labeling
            raise NotImplementedError
            # TODO: update pruning P and labeling L

        # TODO: temporarily assign labels to every leaf and compute error
        L_temp = L.copy()
        raise NotImplementedError

    for i in range(len(P)):
        L = helper.assign_labels(L, P[i], P[i], T, n_samples)

    return L, np.array(error)


def select_case_2(data, labels, T, budget, batch_size):
    """DH algorithm where we choose P by biasing towards choosing nodes in areas where the observed labels are less pure

    :param data: Data matrix 1200x8
    :param labels: true labels 284x1
    :param T: 3 element tree
        T[0] = linkage matrix from hierarchical clustering.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
               for details. If you are unfamiliar with hierarchical clustering using scipy, the following is another helpful resource (We won't use dendrograms
               here, but he gives a nice explanation of how to interpret the linkage matrix):
               https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        T[1] = An array denoting the size of each subtree rooted at node i, where i indexes the array.
               ie. The number of all children + grandchildren + ... + the node itself

        T[2] = dict where keys are nodes and values are the node's parent
    :param budget: Number of iterations to make
    :param batch_size: Number of queries per iteration"""

    n_nodes = len(T[1])  # total nodes in T
    n_samples = len(data)  # total samples in data
    L = np.zeros(n_nodes, dtype=int)  # majority label
    p1 = np.zeros(n_nodes)  # empirical label frequency
    n = np.zeros(n_nodes)  # number of points sampled from each node
    error = []  # np.zeros(n_samples) #error at each round
    root = n_nodes - 1  # corresponds to index of root
    P = np.array([root])
    L[root] = 1

    for i in range(budget):
        selected_P = []
        for b in range(batch_size):
            # TODO: select a node from P biasing towards choosing nodes in areas where the observed labels are less pure
            raise NotImplementedError

            # TODO: pick a random leaf node from subtree Tv and query its label

            # TODO: update empirical counts and probabilities for all nodes u on path from z to v

        for p in selected_P:
            # TODO: update admissible A and compute scores; find best pruning and labeling
            raise NotImplementedError
            # TODO: update pruning P and labeling L

        # TODO: temporarily assign labels to every leaf and compute error
        L_temp = L.copy()
        raise NotImplementedError

    for i in range(len(P)):
        L = helper.assign_labels(L, P[i], P[i], T, n_samples)

    return L, np.array(error)
