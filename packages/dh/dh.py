import numpy as np
import packages.dh.helper as helper
import random


# tested
def _get_proportional_weights(P, T, num_samples):
    """
    Calculates wv = # leaves Tv / # leaves T.

    :param P: array, represents current pruning
    :param T: Tree data structure
    :param num_samples: Number of samples in the data
    :return: |P| x 1 vector, representing selection weights for each subtree v in P
    """

    if len(P) == 1:
        wv = np.array([1.0])  # only node that can be selected
    else:

        # get number of leaves in each subtree in P
        sizes_of_subtree = T[1]
        sizes_of_subtrees_in_P = sizes_of_subtree[P]
        wv = sizes_of_subtrees_in_P / num_samples

    return wv


# tested
def _proportional_selection(P, T, num_samples):
    """
    Select a node v from the pruning, proportional to the size of subtree rooted at v.

    :param P: array, represents current pruning
    :param T: Tree data structure
    :param num_samples: Number of samples in the data
    :return: index of chosen subtree in P
    """
    # set weights wv for each node v in P
    wv = _get_proportional_weights(P, T, num_samples)

    # Each node v in P has a probability of being selected proportional to its weight.
    selected = np.random.choice(P, 1, p=wv)
    return selected[0]


# tested
def _get_confidence_adjusted_weights(P, T, num_samples, n, pHat1):
    """
    Calculates p = wv(1 - p1_LB)

    where
    - wv is the proportional weight of subtree v in P
    - p1_LB is the lower bound of predicted labels over tree T

    :param P: array, represents current pruning
    :param T: Tree data structure
    :param num_samples: Number of samples in the data
    :param n: V x 1 vector, where V is the number of nodes in T.
            n[u] is the number of points sampled from node u
    :param pHat1: V x 1 vector.
            pHat1[u] is the empirical probability of label=1 in tree node u
    :return: |P| x 1 vector, representing selection weights for each subtree v in P
    """
    if len(P) == 1:

        p_scaled = np.array([1.0])  # only node that can be selected
    else:

        # calculate confidence lower bounds
        p0_LB, p1_LB = helper.calculate_confidence_lower_bounds(n, pHat1)

        # limit to nodes in P
        p1_LB_P = p1_LB[P]

        # set weights wv for each node v in P
        wv = _get_proportional_weights(P, T, num_samples)

        # defines probabilities as combination of proportion of dataset and label purity
        p = wv * (1 - p1_LB_P)

        # scale probabilities so they sum to 1
        p_scaled = p / sum(p)

    return p_scaled


# tested
def _confidence_adjusted_selection(P, T, num_samples, n, pHat1):
    """
    Select a node from P biasing towards choosing nodes in areas where the observed labels are less pure.

    :param P: array, represents current pruning
    :param T: Tree data structure
    :param num_samples: Number of samples in the data
    :param n: V x 1 vector, where V is the number of nodes in T.
            n[u] is the number of points sampled from node u
    :param pHat1: V x 1 vector.
            pHat1[u] is the empirical probability of label=1 in tree node u
    :return: index of chosen subtree in P
    """
    p = _get_confidence_adjusted_weights(P, T, num_samples, n, pHat1)
    selected = np.random.choice(P, 1, p=p)

    return selected[0]


# ?? should be set instead of list
def select_case_1(X, y_true, T, budget, batch_size):
    """
    DH algorithm where we choose P proportional to the size of subtree rooted at each node.

    Todo - modularize to make more testable

    :param X: a x b matrix, data samples
    :param y_true: a x 1 vector, true labels for samples
    :param T: Tree data structure
    :param budget: Number of iterations allowed
    :param batch_size: Number of queries per iteration
    :return:
    """

    # define variables
    num_nodes = len(T[1])  # total nodes in T
    num_samples = len(X)  # total samples in data, equal to the number of leaves in T

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
        selected_P = set()  # using set instead of list prevents the same node from being added twice
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
    # assign final labeling based on current pruning
    for j in range(len(P)):
        L = helper.assign_labels(L, P[j], P[j], T, num_samples)  # assign each leaf in Tj the label L(j)
    return L, np.array(error)


def select_case_2(X, y_true, T, budget, batch_size):
    """
    DH algorithm where we choose P by biasing towards choosing nodes in areas where the observed labels are less pure.

    Todo - refactor to eliminate duplication of code from select_case_1

    :param X: a x b matrix, data samples
    :param y_true: a x 1 vector, true labels for samples
    :param T: Tree data structure
    :param budget: Number of iterations allowed
    :param batch_size: Number of queries per iteration
    :return:
    """

    # define variables
    num_nodes = len(T[1])  # total nodes in T
    num_samples = len(X)  # total samples in data, equal to the number of leaves in T

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
        selected_P = set()  # using set instead of list prevents the same node from being added twice
        for b in range(batch_size):
            # TODO: select a node from P biased  towards choosing nodes in areas where the observed labels are less pure
            v = _confidence_adjusted_selection(P, T, num_samples, n, pHat1)
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
    # assign final labeling based on current pruning
    for j in range(len(P)):
        L = helper.assign_labels(L, P[j], P[j], T, num_samples)  # assign each leaf in Tj the label L(j)
    return L, np.array(error)
