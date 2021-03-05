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
    :return: (a x 1 vector, a x b matrix) Tuple where a=len(unlabeled). Represents (f_u, uu_invert).
    """
    # rearrange cells into sub-matrices
    uu = _construct_uu(L, unlabeled)  # a x a matrix
    ul = _construct_ul(L, labeled, unlabeled)  # a x b matrix

    # calculate minimum
    uu_inv = np.linalg.inv(uu)  # a x a matrix
    temp = np.matmul(-1.0 * uu_inv, ul)  # a x b matrix
    minimum = np.matmul(temp, f_l)  # a x 1 vector

    return minimum, uu_inv


# tested
def _add_point_to_f_u(f_u, uu_inv, k, y_k):
    """
    Calculates the updated minimum energy solution for all unlabeled points, if unlabeled point k is given label y_k.
    :param f_u: a x 1 vector where a=len(unlabeled). Minimum energy solution for unlabeled points.
    :param uu_inv: a x a matrix. Inverse matrix of the submatrix of unlabeled points in rearranged Laplacian matrix.
    :param k: scalar, index of one unlabeled point with respect to uu_inv
    :param y_k: scalar, hypothetical label to assign to unlabeled point
    :return: a x 1 vector
    """
    f_k = f_u[k]
    kth_col = uu_inv[:, k]
    kth_diag = uu_inv[k, k]
    inner = (y_k - f_k) * kth_col / kth_diag
    change = inner.reshape(-1, 1)  # reshape to match shape of f_u
    f_u_plus_xk = f_u + change

    return f_u_plus_xk


# tested
def _expected_risk(f_u):
    # for each unlabeled point
    total = 0.0
    for i in range(f_u.shape[0]):
        min_i = min(f_u[i], 1 - f_u[i])
        total += min_i

    return total


# tested
def _expected_estimated_risk(f_u, uu_inv, k):
    """
    Calculates expected risk after querying node k, using the following formula:


    :param uu_inv: Inverse matrix of the submatrix of unlabeled points in the rearranged Laplacian matrix.
    :param k: index of one unlabeled point with respect to uu_inv
    :param f_u: minimum energy solution of all unlabeled points
    :return: scalar, the expected estimated risk
    """

    # expected risk if label y_k = 0
    f_u_plus_xk0 = _add_point_to_f_u(f_u, uu_inv, k, y_k=0)
    Rhat_f_plus_xk0 = _expected_risk(f_u_plus_xk0)

    # expected risk if label y_k = 1
    f_u_plus_xk1 = _add_point_to_f_u(f_u, uu_inv, k, y_k=1)
    Rhat_f_plus_xk1 = _expected_risk(f_u_plus_xk1)

    # estimated expected risk
    f_k = f_u[k]
    Rhat_f_plus_xk = (1 - f_k) * Rhat_f_plus_xk0 + f_k * Rhat_f_plus_xk1

    return Rhat_f_plus_xk


# tested
def zlg_query(f_u, uu_inv, num_labeled, num_samples):
    """
    Chooses a point to labeled so that the expected estimated risk is minimized once the point is added.
    :param f_u: minimum energy solution of all unlabeled points
    :param uu_inv: Inverse matrix of the submatrix of unlabeled points in the rearranged Laplacian matrix.
    :param num_labeled: scalar, number of labeled points
    :param num_samples: scalar, number of samples
    :return: scalar, index of the unlabeled point to query
    """
    query_idx = -1
    min_Rhat = np.inf

    # find the unlabeled point with the minimum expected risk
    num_unlabeled = num_samples - num_labeled
    for k in range(num_unlabeled):
        Rhat = _expected_estimated_risk(f_u, uu_inv, k)

        if Rhat < min_Rhat:
            query_idx = k
            min_Rhat = Rhat

    return query_idx
