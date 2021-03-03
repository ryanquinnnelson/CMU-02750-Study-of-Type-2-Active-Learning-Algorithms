import numpy as np
from scipy.spatial import distance_matrix


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
def _construct_laplacian_matrix(D,W):
    """

    :param D: n x n diagonal matrix
    :param W: n x n weights matrix
    :return: n x n matrix
    """
    return D-W

