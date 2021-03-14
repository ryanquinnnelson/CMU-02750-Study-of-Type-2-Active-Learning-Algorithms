"""
Implementation of binary classifier ZLG, the active learning algorithm from the paper by Zhu, Lafferty, and Ghahramani.
See http://mlg.eng.cam.ac.uk/zoubin/papers/zglactive.pdf.
"""

import numpy as np
import copy
from scipy.spatial import distance_matrix

"""
Note 1 - On the tracking of original labels
The labeled and unlabeled index lists maintained for each ZLG instance are defined relative to X, the entire
data set. This implementation always maintains the labeled samples as the first part of X and unlabeled samples
as the second part of X. 

The query index selected for each round by ZLG is defined relative to the current unlabeled set alone. As 
samples are labeled, they are moved from their arbitrary position in the unlabeled data set to the end of the 
labeled data set. This changes the index position of all unlabeled samples after the chosen sample, relative to the 
original X.
        
        labeled             unlabeled
        [0,1,2,3,4]         [5,6,7,8]  indexes
        [a,b,c,d,e]         [f,g,h,i]  samples
        
        query_idx = 1  // second unlabeled element is the 7th element in original X
        
        labeled             unlabeled
        [0,1,2,3,4,5]       [6,7,8]  indexes
        [a,b,c,d,e,g]       [f,h,i]  samples
        
        query_idx = 1  // second unlabeled element is the 8th element in original X
        
We need a way to determine the original index positions for each queried sample so we can track which unlabeled
samples are selected from X by the ZLG algorithm. If we maintain a separate list of original indexes, we can preserve 
the relationship to X. 
        
        labeled             unlabeled                original
        [0,1,2,3,4]         [5,6,7,8]  indexes       [5,6,7,8]
        [a,b,c,d,e]         [f,g,h,i]  samples
        
        query_idx = 1  // second unlabeled element is the 7th element in original X
        original_idx = 6
        
        labeled             unlabeled                original
        [0,1,2,3,4,5]       [6,7,8]  indexes         [5,7,8]
        [a,b,c,d,e,g]       [f,h,i]  samples
        
        query_idx = 1  // second unlabeled element is the 8th element in original X
        original_idx = 7
"""


# tested
def _calculate_weights_1(X):
    """
    Uses Radial Basis Function (RBF) to calculate weight w_ij for each pair of instances x_i and x_j using the
    formula:

    |
    |  w_ij = exp(  -1 * SUM_d (x_id - x_jd)^2 / σ^2_d )
    |
    |  where
    |  - x_i is a m x 1 vector
    |  - x_id is the d-th dimension of the instance x_i
    |  - σ^2_d is the variance of the d-th column of X
    |  - (x_id - x_jd)^2 is the squared Euclidean distance between x_i and x_j
    |

    Note: This implementation divides each dimension term of a vector pair by the variance of that dimension as part of
    the summation, rather than multiple the entire sum by one variance value.
    See https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf).
    Todo - Find more efficient way to incorporate variance into terms. Really poor performance.
    Todo - Doesn't work correctly. Results in worse performance over time, and I'm not sure why yet.

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


# tested
def _calculate_weights_2(X):
    """
    Uses Radial Basis Function (RBF) to calculate weight w_ij for each pair of instances x_i and x_j using the
    formula:

    |
    |  w_ij = exp(  -1/σ^2 * SUM_d (x_id - x_jd)^2  )
    |
    |  where
    |  - x_i is a m x 1 vector
    |  - x_id is the d-th dimension of the instance x_i
    |  - (x_id - x_jd)^2 is the squared Euclidean distance between x_i and x_j
    |

    Note: This method calculates a single scalar for variance over the entire data.

    :param X: n x m matrix, where n is the number of samples and m is the number of features
    :return: n x n symmetric matrix, where (i,j) is the calculated weight between x_i and x_j
    """

    # compute squared Euclidean distance between each pair of instances
    # X_dist[i][j] is the distance between row i and row j in X
    X_dist_squared = np.square(distance_matrix(X, X))

    # remaining calculations
    a = -1.0 / np.var(X)
    b = a * X_dist_squared
    weights = np.exp(b)

    return weights


# tested
def _construct_weight_matrix(weights, t):
    """
    Constructs n x n graph in which a given pair of nodes is connected by an edge
    only if the precomputed weight w_ij exceeds the user-defined threshold t. If
    the threshold is met, edge (i,j) is created with weight w_ij.

    :param weights: n x n matrix
    :param t: scalar, user-defined threshold for retaining weights
    :return: n x n matrix, where non-zero values represent edges
    """
    W = np.where(weights >= t, weights, 0.0)  # nonzero weight indicates edge
    return W


# tested
def _construct_diagonal_matrix(W):
    """
    Generates diagonal matrix D in which diagonal element d_i is the weighted degree of node Y_i.

    |  d_i = d_(i,i) = SUM_j W(i,j)  // the sum of the ith row of W

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
    # weight matrix
    weights = _calculate_weights_2(X)
    W = _construct_weight_matrix(weights, t)

    # diagonal matrix
    D = _construct_diagonal_matrix(W)

    # Laplacian
    L = _subtract_matrices(D, W)

    return L


# tested
def _build_square_submatrix(L, idx):
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
    Repositions all (i,j) cells from L into a square submatrix:
        Examples
        --------

        L =

        | |  a | ab | ac | ad |
        | | ba |  b | bc | bd |
        | | ca | cb |  c | cd |
        | | da | db | dc |  d |
        |

        Labeled instances: a,d

        _construct_ll(L, [0,3])  # 2x2

        |  |  a | ad |
        |  | da |  d |
        |


        Labeled instances: b,c,d

        _construct_ll(L, [1,2,3])  # 3x3

        |  |  b | bc | bd |
        |  | cb |  c | cd |
        |  | db | dc |  d |
        |

        Labeled instances: d

        _construct_ll(L, [3])  # 1x1

        | | d |
        |

    :param L: n x n matrix, Laplacian
    :param labeled: list of zero-based indexes representing labeled instance positions in the Laplacian matrix.
                    i.e. [0,2] if instance 1 and 3 are labeled.
    :return: b x b matrix, where b is the number of labeled instances
    """
    return _build_square_submatrix(L, labeled)


# tested
def _construct_uu(L, unlabeled):
    """
    Constructs square (unlabeled,unlabeled) instances submatrix from the given Laplacian matrix.

        Examples
        --------

        L =

        | |  a | ab | ac | ad |
        | | ba |  b | bc | bd |
        | | ca | cb |  c | cd |
        | | da | db | dc |  d |
        |

        Unlabeled instances: a,d

        _construct_uu(L, [0,3])  # 2x2

        |  |  a | ad |
        |  | da |  d |
        |


        Unlabeled instances: b,c,d

        _construct_uu(L, [1,2,3])  # 3x3

        |  |  b | bc | bd |
        |  | cb |  c | cd |
        |  | db | dc |  d |
        |

        Unlabeled instances: d

        _construct_uu(L, [3])  # 1x1

        | | d |
        |

    :param L: n x n matrix, Laplacian
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instance 1 and 3 are unlabeled.
    :return: a x a matrix, where a is the number of unlabeled instances
    """
    return _build_square_submatrix(L, unlabeled)


# tested
def _build_rectangular_submatrix(L, idx_i, idx_j):
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

        Examples
        --------

        L =

        | |  a | ab | ac | ad |
        | | ba |  b | bc | bd |
        | | ca | cb |  c | cd |
        | | da | db | dc |  d |
        |

        Labeled instances: a,d
        Unlabeled instances: b,c

        _construct_lu(L, [0,3], [1,2])  # 2x2

        |  | ab | ac |
        |  | db | dc |
        |

        Labeled instances: a
        Unlabeled instances: b,c,d

        _construct_lu(L, [0], [1,2,3])  # 1x3

        |  | ab | ac | ad |
        |

        Labeled instances: a,b,c
        Unlabeled instances: d

        _construct_lu(L, [0,1,2], [3])  # 3x1

        | | ad |
        | | bd |
        | | cd |
        |



    :param L: n x n matrix, Laplacian
    :param labeled: list of indexes representing labeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are labeled.
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are unlabeled.
    :return: b x a matrix, where b is the number of labeled instances and a is the number of unlabeled
    """
    return _build_rectangular_submatrix(L, labeled, unlabeled)


# tested
def _construct_ul(L, labeled, unlabeled):
    """
    Constructs rectangular (unlabeled,labeled) instances submatrix from the given Laplacian matrix.

        Examples
        --------

        L =

        | |  a | ab | ac | ad |
        | | ba |  b | bc | bd |
        | | ca | cb |  c | cd |
        | | da | db | dc |  d |
        |

        Labeled instances: a,d
        Unlabeled instances: b,c

        _construct_ul(L, [1,2], [0,3])  # 2x2

        |  | ba | bd |
        |  | ca | cd |
        |

        Labeled instances: a
        Unlabeled instances: b,c,d

        _construct_ul(L, [1,2,3], [0])  # 3x1

        | | ba |
        | | ca |
        | | da |
        |


        Labeled instances: a,b,c
        Unlabeled instances: d

        _construct_ul(L, [3], [0,1,2])  # 1x3

        | | da | db | dc |
        |

    :param L: n x n matrix, Laplacian
    :param labeled: list of indexes representing labeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are labeled.
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are unlabeled.
    :return: a x b matrix, where b is the number of labeled instances and a is the number of unlabeled
    """
    return _build_rectangular_submatrix(L, unlabeled, labeled)


# tested
def _rearrange_laplacian_matrix(L, labeled, unlabeled):
    """
    Rearranges the cells of the Laplacian matrix by grouping labeled and unlabeled instances into submatrices and
    combining the submatrices back together in a specific pattern. See subroutines for additional details.

    |  | ll | lu |
    |  | ul | uu |
    |
    |  where
    |  - ll are the (labeled,labeled) instances
    |  - uu are the (unlabeled,unlabeled) instances
    |  - lu are the (labeled, unlabeled) instances
    |  - ul are the (unlabeled, labeled) instances

    :param L: n x n matrix, Laplacian
    :param labeled: list of indexes representing labeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are labeled.
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are unlabeled.
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
    Calculates minimum energy solution f_u for all unlabeled instances given the labeled instances.
    Uses the following formula:

    |
    |  f_u = -1 * uu_inv * ul * f_l
    |
    |  where
    |  - uu_inv is the inverted (unlabeled, unlabeled) submatrix generated from the Laplacian.
    |  - ul is the (unlabeled, labeled) submatrix generated from the Laplacian.
    |  - f_u is the mean values on unlabeled data.
    |

    The minimum energy solution for the kth unlabeled instance is the estimated binary label on [0,1].
    We infer the label by rounding: i.e. f_k = 0.1 would be rounded to 0 and f_k = 0.9 would be rounded to 1.

    :param L: n x n matrix, Laplacian
    :param labeled: list of indexes representing labeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are labeled.
    :param unlabeled: list of indexes representing unlabeled instance positions in the Laplacian matrix.
                        i.e. [0,2] if instances 1 and 3 are unlabeled.
    :param f_l: b x 1 vector of labeled instances, where b is the number of labeled instances.
    :return: (a x 1 vector, a x b matrix) Tuple which represents (f_u, uu_inv).
    """
    # rearrange cells into submatrices
    uu = _construct_uu(L, unlabeled)  # a x a matrix
    ul = _construct_ul(L, labeled, unlabeled)  # a x b matrix

    # calculate minimum solution
    uu_inv = np.linalg.inv(uu)  # a x a matrix
    temp = np.matmul(-1.0 * uu_inv, ul)  # a x b matrix
    f_u = np.matmul(temp, f_l)  # a x 1 vector

    return f_u, uu_inv


def _update_minimum_energy_solution(f_u, uu_inv, k, y_k):
    """
    Calculates updated minimum energy solution for all unlabeled points if unlabeled point k is given label y_k.
    Uses the following formula:

    |
    |  f_u_plus_xk = f_u + (y_k - f_k) * uu_inv_{.k} * 1/uu_inv_{kk}
    |
    |  where
    |  - f_u is the updated minimum energy solution of unlabeled points if instance x_k was labeled with y_k
    |  - f_k is the current minimum energy solution of the kth unlabeled point
    |  - uu_inv_{.k} is the kth column of the inverse Laplacian on unlabeled data
    |  - uu_inv_{kk} is the kth diagonal element of the same matrix
    |

    Todo - if statement needs testing
    Todo - Need to understand why some diagonals are zero. Is it a rounding error or bug somewhere else?

    :param f_u: a x 1 vector, where a is the number of unlabeled instances.
                Represents the minimum energy solution for unlabeled points before adding point k.
    :param uu_inv: a x a matrix, inverse matrix of the submatrix of unlabeled points
                from the rearranged Laplacian matrix.
    :param k: scalar, index of one unlabeled instance with respect to uu_inv
    :param y_k: scalar, hypothetical label to assign to unlabeled instance k
    :return: a x 1 vector, representing the updated minimum energy solution
    """

    f_k = f_u[k]
    kth_col = uu_inv[:, k]
    kth_diag = uu_inv[k, k]

    if kth_diag == 0:  # unexpected issue
        f_u_plus_xk = f_u  # no change as temporary workaround
    else:
        change = (y_k - f_k) * kth_col / kth_diag
        f_u_plus_xk = f_u + change

    return f_u_plus_xk


# tested
def _expected_risk(f_u):
    """
    Calculates the expected risk of all unlabeled instances in the given minimum energy solution.
    Uses the following formula:

    |
    |  Rhat(f_u) = SUM_i  min{  f_i, 1 - f_i  }
    |
    |  where
    |  - i is the ith unlabeled instance in vector f_u (out of n)
    |  - f_i is the minimum energy solution for the ith unlabeled instance

    :param f_u: a x 1 vector where a is the number of unlabeled instances.
                Represents the minimum energy solution for unlabeled points.
    :return: scalar
    """

    total = 0.0

    # add minimum energy for each unlabeled point in f_u
    for i in range(f_u.shape[0]):
        min_i = min(f_u[i], 1 - f_u[i])
        total += min_i

    return total


# tested
def expected_estimated_risk(f_u, uu_inv, k):
    """
    Calculates expected risk after querying node k. Uses the following formula:

    |
    |  Rhat(f_u_plus_xk) = (1 - f_k) * Rhat(f_u_plus_xk0) + f_k * Rhat(f_u_plus_xk1)
    |
    |  where
    |  - f_u_plus_xk is the updated minimum energy solution of unlabeled points if instance x_k was labeled
    |  - f_k is the current minimum energy solution of the kth unlabeled point
    |  - f_u_plus_xk0 is the updated minimum energy solution of unlabeled points if instance x_k was labeled y_k=0
    |  - f_u_plus_xk1 is the updated minimum energy solution of unlabeled points if instance x_k was labeled y_k=1

    :param uu_inv: Inverse matrix of the submatrix of unlabeled points in the rearranged Laplacian matrix.
    :param k: index of one unlabeled point with respect to uu_inv
    :param f_u: minimum energy solution of all unlabeled points
    :return: scalar, the expected estimated risk
    """

    # expected risk if label y_k = 0
    f_u_plus_xk0 = _update_minimum_energy_solution(f_u, uu_inv, k, y_k=0)
    Rhat_f_plus_xk0 = _expected_risk(f_u_plus_xk0)

    # expected risk if label y_k = 1
    f_u_plus_xk1 = _update_minimum_energy_solution(f_u, uu_inv, k, y_k=1)
    Rhat_f_plus_xk1 = _expected_risk(f_u_plus_xk1)

    # estimated expected risk of getting any label for point k
    f_k = f_u[k]
    Rhat_f_plus_xk = (1.0 - f_k) * Rhat_f_plus_xk0 + f_k * Rhat_f_plus_xk1

    return Rhat_f_plus_xk


# tested
def zlg_query(f_u, uu_inv, num_labeled, num_samples):
    """
    Chooses an instance to label such that the expected estimated risk of the resulting model is minimized.

    :param f_u: a x 1 vector where a is the number of unlabeled instances.
                Represents the minimum energy solution for unlabeled points.
    :param uu_inv: a x a matrix. Inverse matrix of the submatrix of unlabeled points
                    from the rearranged Laplacian matrix.
    :param num_labeled: scalar, number of labeled points
    :param num_samples: scalar, number of instances
    :return: scalar, index of the unlabeled point to query relative to f_u
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


class ZLG:

    # tested
    def __init__(self, Xk, yk, Xu, yu):
        """
        Initializes instance and sets tracking for labeled and unlabeled samples.

        :param Xk: labeled samples
        :param yk: labels for labeled samples
        :param Xu: unlabeled samples
        :param yu: labels for unlabeled samples
        """
        self.Xk = Xk
        self.yk = yk
        self.Xu = Xu
        self.yu = yu

        # track indices of labeled and unlabeled
        self.labeled = [i for i in range(len(yk))]
        self.unlabeled = [i for i in range(len(yk), len(Xk) + len(Xu))]

        # maintain predictions for unlabeled samples
        self.fu = None

    # tested
    def _update_sets(self, query_idx):
        """
        Moves selected sample from arbitrary position in unlabeled data set to end of labeled data set.
        :param query_idx:
        :return: None
        """
        # add instance to end of labeled set
        self.yk = np.append(self.yk, self.yu[query_idx])
        self.Xk = np.append(self.Xk, [self.Xu[query_idx, :]], axis=0)

        # remove instance from unlabeled set
        self.yu = np.delete(self.yu, query_idx)
        self.Xu = np.delete(self.Xu, query_idx, axis=0)

    # tested
    def _update_indexes(self):
        """
        Moves the first element in list of unlabeled indexes to end of the list of labeled indexes.
        :return: None
        """
        self.labeled.append(self.unlabeled.pop(0))

    # tested
    def score(self):
        """
        Calculates the accuracy of predicted labels.
        :return: float, calculated accuracy
        """

        y_pred = np.round(self.fu)
        y_true = self.yu

        wrong = (y_pred != y_true).sum()
        error = wrong / len(y_true)
        return 1.0 - error

    # tested
    def improve_predictions(self, t, budget):
        """
        Uses the sampling budget and user-defined similarity threshold to improve the predictions of unlabeled
        samples in the data pool.

        :param t: float, similarity threshold
        :param budget: number of queries allowed to the oracle
        :return: (list, list) Tuple representing (queried_indexes, scores) where
                queried_indexes is the list of indexes queried each round (relative to the original X), and
                scores is the accuracy of the predicted labels for remaining unlabeled data after labeling the
                sample selected for that round.
        """
        # track original indices of unlabeled samples
        original_indexes = copy.deepcopy(self.unlabeled)  # see "Note 1 - On the tracking of original labels"

        # initialize components
        X = np.concatenate((self.Xk, self.Xu), axis=0)
        delta = laplacian_matrix(X, t)
        self.fu, delta_uu_inv = minimum_energy_solution(delta, self.labeled, self.unlabeled, self.yk)

        # use query budget to improve predictions
        queried_indexes = []
        scores = []
        for query in range(budget):
            # select unlabeled sample to query
            query_idx = zlg_query(self.fu, delta_uu_inv, len(self.yk), len(X))

            # record which sample was queried
            original_idx = original_indexes[query_idx]  # get index relative to original X
            queried_indexes.append(original_idx)
            original_indexes.pop(query_idx)  # remove element from same relative position to preserve original indexing

            # update fields
            self._update_indexes()
            self._update_sets(query_idx)  # move newly labeled sample to labeled set

            # update Laplacian
            X = np.concatenate((self.Xk, self.Xu), axis=0)
            delta = laplacian_matrix(X, t)

            # update label predictions given newly labeled sample
            self.fu, delta_uu_inv = minimum_energy_solution(delta, self.labeled, self.unlabeled, self.yk)

            # score predictions
            y_pred = np.round(self.fu)
            scores.append(self.score())

        return queried_indexes, scores
