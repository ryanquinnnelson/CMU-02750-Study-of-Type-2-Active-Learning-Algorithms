import numpy as np
from scipy.spatial import distance_matrix
import itertools


# ?? how to calculate variance of X
def _calculate_weights(X):
    """
    Uses Radial Basis Function to calculate weight w_ij for each pair of instances x_i and x_j.
    :param X: n x m matrix, where n is the number of samples and m is the number of features
    :return: n x n matrix
    """
    # calculate variance of X for each of its m dimensions
    x_var = np.var(X, axis=1)  # 1 x m vector
    variance = 1.0

    # compute squared Euclidean distance between each pair of instances
    # distances[i][j] is the distance between record i and record j
    X_dist = distance_matrix(X, X)
    X_dist_squared = np.square(X_dist)

    # remaining calculations
    a = -1 / variance
    inner = np.multiply(X_dist_squared, a)
    weights = np.exp(inner)

    return weights


# tested
def _construct_weight_matrix(weights, t):
    """
    Retain values where weight w_ij >= t. Replace values where weight w_ij < t with 0.0
    :param weights: n x n matrix
    :param t: scalar threshold
    :return: n x n matrix
    """
    W = np.where(weights >= t, weights, 0.0)
    return W


# tested
def _construct_diagonal_matrix(W):
    """
    Generates a diagonal matrix D=diag(d_i) where d_i = SUM_j w_ij.

    :param W: n x n matrix
    :return: n x n matrix
    """
    n = W.shape[0]

    W_sums = np.sum(W, axis=1)  # sum each row
    D = np.zeros((n, n))
    np.fill_diagonal(D, W_sums)

    return D


# tested
def _construct_laplacian_matrix(D, W):
    """

    :param D: n x n diagonal matrix
    :param W: n x n weights matrix
    :return: n x n matrix
    """
    return D - W


# tested
def _construct_ll(L, labeled):
    """
    Assumes labeled is sorted in ascending order.
    Assumes labeled list is not empty.
    :param L: n x n matrix
    :param labeled: sorted list of indices of unlabeled instances
    :return: b x b matrix, where b=len(labeled)
    """

    num_labeled = len(labeled)

    # build ll scaffold
    ll = np.zeros((num_labeled, num_labeled))

    # generate all combinations of labeled indices i.e. (1,2) -> (1,1), (1,2), (2,1), (2,2)
    for idx1, i in enumerate(labeled):
        for idx2, j in enumerate(labeled):
            ll[idx1][idx2] = L[i][j]

    return ll


# tested
def _construct_uu(L, unlabeled):
    """
    Todo - this is the same code as construct_ll(). Combine the two.
    :param L: n x n matrix
    :param unlabeled: sorted list of indices of unlabeled instances
    :return: a x a matrix, where a=len(unlabeled)
    """
    num_unlabeled = len(unlabeled)

    # build uu scaffold
    uu = np.zeros((num_unlabeled, num_unlabeled))

    # generate all combinations of unlabeled indices i.e. (1,2) -> (1,1), (1,2), (2,1), (2,2)
    for idx1, i in enumerate(unlabeled):
        for idx2, j in enumerate(unlabeled):
            uu[idx1][idx2] = L[i][j]

    return uu


# tested
def _construct_lu(L, labeled, unlabeled):
    """

    :param L: n x n matrix
    :param labeled: sorted list of indices of labeled instances i.e. [0,2] if instance 1 and 3 are labeled.
    :param unlabeled: sorted list of indices of unlabeled instances
    :return: b x a matrix, where b=len(labeled) and a=len(unlabeled)
    """
    num_labeled = len(labeled)
    num_unlabeled = len(unlabeled)

    # build lu scaffold
    lu = np.zeros((num_labeled, num_unlabeled))

    # generate
    for idx1, i in enumerate(labeled):
        for idx2, j in enumerate(unlabeled):
            lu[idx1][idx2] = L[i][j]

    return lu


# tested
def _construct_ul(L, labeled, unlabeled):
    """

    :param L: n x n matrix
    :param labeled: sorted list of indices of labeled instances i.e. [0,2] if instance 1 and 3 are labeled.
    :param unlabeled: sorted list of indices of unlabeled instances
    :return: a x b matrix, where a=len(unlabeled) and b=len(labeled)
    """
    num_labeled = len(labeled)
    num_unlabeled = len(unlabeled)

    # build ul scaffold
    ul = np.zeros((num_unlabeled, num_labeled))

    # generate
    for idx1, i in enumerate(unlabeled):
        for idx2, j in enumerate(labeled):
            ul[idx1][idx2] = L[i][j]

    return ul


# tested
def _rearrange_laplacian_matrix(L, labeled, unlabeled):
    """
    Rearranges the cells of the matrix by grouping labeled and unlabeled instances together in a specific pattern.

    ll  lu
    ul  uu

    where
    - ll are the labeled instances
    - uu are the unlabeled instances
    - lu are the pairs of (labeled, unlabeled) instances
    - ul are the pairs of (unlabeled, labeled) instances

    :param L: n x n matrix
    :param labeled: sorted list of indices of labeled instances i.e. [0,2] if instance 1 and 3 are labeled.
    :param unlabeled: sorted list of indices of unlabeled instances
    :return: n x n matrix
    """
    # rearrange cells into sub-matrices
    ll = _construct_ll(L, labeled)
    uu = _construct_uu(L, unlabeled)
    lu = _construct_lu(L, labeled, unlabeled)
    ul = _construct_ul(L, labeled, unlabeled)

    # combine blocks back together
    combined = np.block([[ll, lu],
                         [ul, uu]])
    return combined


# tested
def _calc_minimum_energy_solution(L, labeled, unlabeled, f_l):
    """
    Calculates f_u.

    :param L: n x n matrix
    :param labeled:
    :param unlabeled:
    :param f_l: b x 1 vector of labeled instances, where b=len(labeled)
    :return: a x 1 vector, where a=len(unlabeled)
    """
    # rearrange cells into sub-matrices
    uu = _construct_uu(L, unlabeled)  # a x a matrix
    ul = _construct_ul(L, labeled, unlabeled)  # a x b matrix

    # calculate minimum
    uu_invert = np.linalg.inv(uu)
    temp = np.matmul(-1.0 * uu_invert, ul)
    minimum = np.matmul(temp, f_l)

    return minimum
