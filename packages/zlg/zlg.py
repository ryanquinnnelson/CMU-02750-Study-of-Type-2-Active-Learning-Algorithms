import numpy as np
from scipy.spatial import distance_matrix


# ?? how to calculate variance of X
def _calculate_weights(X):
    """
    Uses Radial Basis Function to calculate weight w_ij for each pair of instances x_i and x_j.
    :param X: n x m matrix, where n is the number of samples and m is the number of features
    :return:
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


def _construct_weight_matrix(weights, t):
    """
    Retain values where weight w_ij >= t. Replace values where weight w_ij < t with 0.0
    :param weights:
    :param t:
    :return:
    """
    W = np.where(weights >= t, weights, 0.0)
    return W
