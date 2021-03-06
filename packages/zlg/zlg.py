"""
Implementation of ZLG algorithm from the paper by Zhu, Lafferty, and Ghahramani.
See http://mlg.eng.cam.ac.uk/zoubin/papers/zglactive.pdf.
"""

import numpy as np
from scipy.spatial import distance_matrix


# ?? confirm squared Euclidean distance
# tested
def _calculate_weights_1(X):
    """
    Uses Radial Basis Function (RBF) to calculate weight w_ij for each pair of instances x_i and x_j using the
    formula:

    w_ij = exp(  -1 * SUM_d (x_id - x_jd)^2 / σ^2_d )

    Note: This implementation divides each dimension term of a vector pair by the variance of that dimension as part of
    the summation, rather than multiple the entire sum by one variance value.
    Todo - Find more efficient way to calculate this. Currently O(n^3).
    :param X: n x m matrix, where n is the number of samples and m is the number of features
    :return: n x n matrix
    """

    # calculate variance of X for each of its m dimensions
    X_var = np.var(X, axis=0)  # m x 1 vector

    # perform summation for each pair of instances
    rows = X.shape[0]
    cols = X.shape[1]
    totals = np.zeros((rows, rows))  # matrix to hold totals as they are calculated
    for i in range(rows):
        for j in range(rows):

            total = 0.0
            for d in range(cols):
                total += (X[i][d] - X[j][d]) * (X[i][d] - X[j][d]) / X_var[d]

            totals[i][j] = total

    # remaining calculations
    weights = np.exp(-1.0 * totals)

    return weights


# ?? confirm squared Euclidean distance
# tested
def _calculate_weights_2(X):
    """
    Uses Radial Basis Function (RBF) to calculate weight w_ij for each pair of instances x_i and x_j using the
    formula:

    w_ij = exp(  -1/σ^2 * SUM_d (x_id - x_jd)^2  )

    Note: This method calculates a single scalar for variance over the entire data.

    :param X: n x m matrix, where n is the number of samples and m is the number of features
    :return: n x n symmetric matrix, where (i,j) is the calculated weight between x_i and x_j
    """

    # compute squared Euclidean distance between each pair of instances
    # X_dist[i][j] is the distance between row i and row j in X
    X_dist_squared = np.square(distance_matrix(X, X))

    # remaining calculations
    a = -1.0 / np.var(X)
    b = np.multiply(X_dist_squared, a)
    weights = np.exp(b)

    return weights


# tested
def _construct_weight_matrix(weights, t):
    """
    Filters out weights w_ij < t and replaces values with zero.

    :param weights: n x n matrix
    :param t: scalar, user-defined threshold for retaining weights
    :return: n x n matrix
    """
    W = np.where(weights >= t, weights, 0.0)
    return W


# tested
def _construct_diagonal_matrix(W):
    """
    Generates diagonal matrix D in which diagonal element d_i is equal to the sum of the ith row of W.

    :param W: n x n matrix
    :return: n x n matrix
    """
    n = W.shape[0]

    W_sums = np.sum(W, axis=1)  # sum each row
    D = np.zeros((n, n))
    np.fill_diagonal(D, W_sums)

    return D


# tested
def _subtract_matrices(D, W):
    """
    Subtracts weights matrix W from diagonal matrix D.

    :param D: n x n diagonal matrix
    :param W: n x n weights matrix
    :return: n x n matrix
    """
    return D - W


# tested
def laplacian_matrix(X, t):
    """
    Performs all steps needed to derive the combinatorial Laplacian matrix DELTA from the data.

    :param X: n x m matrix, where n is the number of instances and m is the number of features
    :param t: scalar, user-defined threshold for retaining weights
    :return: n x n matrix
    """
    # calculate weight matrix W
    weights = _calculate_weights_2(X)  # faster than method 1
    W = _construct_weight_matrix(weights, t)
    D = _construct_diagonal_matrix(W)
    L = _subtract_matrices(D, W)

    return L


# tested
def _construct_square_submatrix(L, idx):
    """
    Constructs square submatrix from the given Laplacian matrix. The process for submatrix ll and uu is exactly the
    same, only the list is different.
    Todo - Determine if sorting really needs to be performed.
    :param L: n x n matrix, Laplacian
    :param idx: list of zero-based indexes representing instance positions in the Laplacian matrix.
                    i.e. [0,2] if instance 1 and 3 are selected.
    :return: c x c matrix, where c is the number of indexes provided
    """
    # sort labeled
    idx_sorted = sorted(idx)

    # scaffold for values
    c = len(idx_sorted)
    scaffold = np.zeros((c, c))

    # all combinations of indexes i.e. 1 and 2 -> (1,1), (1,2), (2,1), (2,2)
    for idx1, i in enumerate(idx_sorted):
        for idx2, j in enumerate(idx_sorted):
            scaffold[idx1][idx2] = L[i][j]

    return scaffold


# tested
def _construct_ll(L, labeled):
    """
    Constructs square (labeled,labeled) submatrix from the given Laplacian matrix.

    :param L: n x n matrix, Laplacian
    :param labeled: list of zero-based indexes representing labeled instance positions in the Laplacian matrix.
                    i.e. [0,2] if instance 1 and 3 are labeled.
    :return: b x b matrix, where b is the number of labeled instances
    """
    return _construct_square_submatrix(L, labeled)


# tested
def _construct_uu(L, unlabeled):
    """
    Constructs square (unlabeled,unlabeled) instances submatrix from the given Laplacian matrix.

    :param L: n x n matrix, Laplacian
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instance 1 and 3 are unlabeled.
    :return: a x a matrix, where a is the number of unlabeled instances
    """
    return _construct_square_submatrix(L, unlabeled)


def _construct_rectangular_submatrix(L, idx_i, idx_j):
    """
    Constructs rectangular submatrix from the given Laplacian matrix.
    Todo - Determine if sorting is really necessary.
    :param L: n x n matrix, Laplacian
    :param idx_i: list of indexes representing instance positions in the Laplacian matrix.
                        i.e. [0,2] if instance 1 and 3 are selected.
                    Defines the rows in the submatrix.
    :param idx_j: list of indexes representing instance positions in the Laplacian matrix.
                        i.e. [0,2] if instance 1 and 3 are selected.
                        Defines the columns in the submatrix.
    :return: i x j matrix, where i is the length of idx_i and j is the length of idx_j
    """
    # sort lists
    idx_i_sorted = sorted(idx_i)
    idx_j_sorted = sorted(idx_j)

    # build scaffold
    num_i = len(idx_i_sorted)
    num_j = len(idx_j_sorted)
    scaffold = np.zeros((num_i, num_j))

    # all combinations of indexes
    for idx1, i in enumerate(idx_i_sorted):
        for idx2, j in enumerate(idx_j_sorted):
            scaffold[idx1][idx2] = L[i][j]

    return scaffold


# tested
def _construct_lu(L, labeled, unlabeled):
    """
    Constructs rectangular (labeled,unlabeled) instances submatrix from the given Laplacian matrix.

    :param L: n x n matrix, Laplacian
    :param labeled: list of indexes representing labeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are labeled.
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are unlabeled.
    :return: b x a matrix, where b is the number of labeled instances and a is the number of unlabeled
    """
    return _construct_rectangular_submatrix(L, labeled, unlabeled)


# tested
def _construct_ul(L, labeled, unlabeled):
    """
    Constructs rectangular (unlabeled,labeled) instances submatrix from the given Laplacian matrix.

    :param L: n x n matrix, Laplacian
    :param labeled: list of indexes representing labeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are labeled.
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are unlabeled.
    :return: a x b matrix, where b is the number of labeled instances and a is the number of unlabeled
    """
    _construct_rectangular_submatrix(L, unlabeled, labeled)



# tested
def _rearrange_laplacian_matrix(L, labeled, unlabeled):
    """
    Rearranges the cells of the matrix by grouping labeled and unlabeled instances together in a specific pattern.

    ll  lu
    ul  uu

    where
    - ll are the (labeled,labeled) instances
    - uu are the (unlabeled,unlabeled) instances
    - lu are the (labeled, unlabeled) instances
    - ul are the (unlabeled, labeled) instances

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
def minimum_energy_solution(L, labeled, unlabeled, f_l):
    """
    Calculates minimum energy solution f_u for all unlabeled instances.

    :param L: n x n matrix
    :param labeled: sorted list of indices of labeled instances i.e. [0,2] if instance 1 and 3 are labeled.
    :param unlabeled: sorted list of indices of unlabeled instances
    :param f_l: b x 1 vector of labeled instances, where b is the number of labeled instances
    :return: (a x 1 vector, a x b matrix) Tuple where a is the number of unlabeled instances.
            Tuple represents (f_u, uu_inv).
    """
    # rearrange cells into sub-matrices
    uu = _construct_uu(L, unlabeled)  # a x a matrix
    ul = _construct_ul(L, labeled, unlabeled)  # a x b matrix

    # calculate minimum
    uu_inv = np.linalg.inv(uu)  # a x a matrix
    temp = np.matmul(-1.0 * uu_inv, ul)  # a x b matrix
    f_u = np.matmul(temp, f_l)  # a x 1 vector

    return f_u, uu_inv


def _add_point_to_f_u(f_u, uu_inv, k, y_k):
    """
    Calculates updated minimum energy solution for all unlabeled points, if unlabeled point k is given label y_k.
    Todo - if statement needs testing
    :param f_u: a x 1 vector where a is the number of unlabeled instances.
                Represents the minimum energy solution for unlabeled points.
    :param uu_inv: a x a matrix.
                    Inverse matrix of the submatrix of unlabeled points from the rearranged Laplacian matrix.
    :param k: scalar, index of one unlabeled instance with respect to uu_inv
    :param y_k: scalar, hypothetical label to assign to unlabeled instance
    :return: a x 1 vector, representing the updated minimum energy solution
    """

    f_k = f_u[k]
    kth_col = uu_inv[:, k]
    kth_diag = uu_inv[k, k]

    if kth_diag == 0:
        f_u_plus_xk = f_u  # no change
    else:
        change = (y_k - f_k) * kth_col / kth_diag
        f_u_plus_xk = f_u + change

    return f_u_plus_xk


# tested
def _expected_risk(f_u):
    """
    Calculates the expected risk of all unlabeled instances in the given minimum energy solution.
    :param f_u: a x 1 vector where a is the number of unlabeled instances.
                Represents the minimum energy solution for unlabeled points.
    :return: scalar
    """
    # for each unlabeled point
    total = 0.0
    for i in range(f_u.shape[0]):
        min_i = min(f_u[i], 1 - f_u[i])
        total += min_i

    return total


# tested
def expected_estimated_risk(f_u, uu_inv, k):
    """
    Calculates expected risk after querying node k.

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
    Chooses an instance to label such that the expected estimated risk of the resulting model is minimized.
    :param f_u: a x 1 vector where a is the number of unlabeled instances.
                Represents the minimum energy solution for unlabeled points.
    :param uu_inv: a x a matrix.
                    Inverse matrix of the submatrix of unlabeled points from the rearranged Laplacian matrix.
    :param num_labeled: scalar, number of labeled points
    :param num_samples: scalar, number of samples
    :return: scalar, index of the unlabeled point to query
    """
    query_idx = -1
    min_Rhat = np.inf

    # find the unlabeled point with the minimum expected risk
    num_unlabeled = num_samples - num_labeled
    for k in range(num_unlabeled):

        Rhat = expected_estimated_risk(f_u, uu_inv, k)
        if Rhat < min_Rhat:
            query_idx = k
            min_Rhat = Rhat

    return query_idx
