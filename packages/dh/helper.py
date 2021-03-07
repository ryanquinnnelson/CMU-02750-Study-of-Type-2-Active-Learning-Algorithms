"""
Code provided to help implementation of DH, modified to be modular and testable.
"""

import numpy as np

from scipy.cluster.hierarchy import linkage

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim


# tested
def generate_T(X):
    """
    Builds data structure T to contain linkage, subtree sizes, and subtree dictionary.

    |  T is a 3-element tree consisting of the following:
    |  - T[0] = linkage matrix from hierarchical clustering.
    |  - T[1] = An array denoting the size of each subtree rooted at node i, where i indexes the array.
        i.e. The number of all leaves in subtree rooted at node i (w_i in the paper).
    |  - T[2] = dict where keys are nodes and values are the node's parent.

    |
    Note 1: If hierarchical clustering is performed on 4 samples, the total number of nodes represented by Z is 5.
    This is because the final clustering is not recorded because it is redundant:

    |
    |  Given : 0  1  2  3
    |  Step 1 combine 2 and 3 to get node 4   // 3 nodes in hierarchy
    |  Step 2 combine 1 and 4 to get node 5   // 5 nodes in hierarchy
    |  Stop - sample 0 is not given a node because we know the final clustering is 0 and 5.
    Hierarchical clustering does not record combining 0 and 5 to get 6, so the overall tree is missing 2 nodes.

    Todo - set root node parent as -1 instead of 0 to avoid confusion

    :param X: a x b matrix, data to be clustered
    :return: Tree data structure
    """

    # perform hierarchical clustering over training set
    Z = linkage(X, method='ward')

    # build (a-1) x 2 linkage matrix where n is the number of instances in the training set
    # row i contains ids of nodes combined at ith iteration
    link = Z[:, :2].astype(int)

    # define subtrees for each node in the hierarchy
    num_samples = len(X)
    num_nodes = link[-1, -1]  # the number of nodes generated via clustering - See Note 1
    num_subtrees = num_nodes + 2  # add two additional nodes to account for final clustering - See Note 1
    subtree_sizes = np.zeros(num_subtrees)  # scaffold to list number of leaves per subtree
    subtree_sizes[:num_samples] = 1  # subtrees containing an individual sample before clustering have a single leaf

    # link subtrees together and use dictionary to define relationships between nodes
    # ?? shouldn't root be set to -1 or something not representing another node in the tree - we have a node 0
    parents = {2 * (num_samples - 1): 0}  # dictionary relating subtrees together, with root of tree set to 0
    for i in range(len(link)):
        left_child = link[i, 0]
        right_child = link[i, 1]
        parent = i + num_samples
        parents[left_child] = parent
        parents[right_child] = parent

        # calculates number of leaves in subtree rooted at this parent
        subtree_sizes[parent] = subtree_sizes[left_child] + subtree_sizes[right_child]

    # build data structure to access all components
    T = [link, subtree_sizes, parents]

    return T


def get_classifier(choice, X=None, y=None, seed=None):
    """
    Generates a classifier of the specified choice.

    :param choice: String representing the desired classifier
    :param X: Used to define Neural Network
    :param y: Used to define Neural Network
    :param seed: Used to define Random Forest, Gradient Boosting, Neural Network
    :return: initialized model
    """
    model = None

    if choice == 'Logistic Regression':
        lr = LogisticRegression()
        model = lr

    elif choice == 'Random Forest':
        N_estimator_rf = 20
        MAX_depth_rf = 6
        rf = RandomForestClassifier(n_estimators=N_estimator_rf,
                                    max_depth=MAX_depth_rf,
                                    random_state=seed)
        model = rf

    elif choice == 'Gradient Boosting Decision Tree':
        N_estimator_gbdt = 20
        gbdt_max_depth = 6
        gbdt = GradientBoostingClassifier(n_estimators=N_estimator_gbdt,
                                          learning_rate=0.1,
                                          max_depth=gbdt_max_depth,
                                          random_state=seed)
        model = gbdt

    elif choice == 'Neural Net':

        # 3-layer fully connected neural network
        torch.manual_seed(seed)

        class NNClassifier(object):
            def __init__(self,
                         feature_n,
                         class_n,
                         hidden_n=30,
                         learning_rate=4e-3,
                         weight_decay=1e-5):
                self.model = torch.nn.Sequential(torch.nn.Linear(feature_n, hidden_n),
                                                 torch.nn.SiLU(),
                                                 torch.nn.Linear(hidden_n, hidden_n),
                                                 torch.nn.SiLU(),
                                                 torch.nn.Linear(hidden_n, class_n))
                self.lr = learning_rate
                self.wd = weight_decay

            def fit(self, X_train, y_train, epoches=300, batch_size=50):
                X_t = torch.from_numpy(X_train.astype(np.float32))
                y_t = torch.from_numpy(y_train.astype(np.int64))
                dataset = TensorDataset(X_t, y_t)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
                loss_record = 0.0
                report_epoch = 50
                for epoch_i in range(epoches):
                    for batch in loader:
                        x_batch, y_batch = batch
                        y_pred = self.model(x_batch)
                        loss = loss_fn(y_pred, y_batch)
                        self.model.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_record += loss.item()
                    if epoch_i % report_epoch == report_epoch - 1:
                        # print("[%d|%d] epoch loss:%.2f" % (epoch_i + 1, epoches, loss_record / report_epoch))
                        loss_record = 0.0
                    if epoch_i >= epoches:
                        break
                return self

            def score(self, X_test, y_test):
                X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
                y_pred_test = self.model(X_test_tensor)
                y_output = torch.argmax(y_pred_test, axis=1).numpy()
                return (y_output == y_test).mean()

        nn = NNClassifier(feature_n=X.shape[1], class_n=len(np.unique(y)))
        model = nn

    return model


# tested
def compute_error(y_pred, y_true):
    """
    Computes the percent of mislabeled instances out of n samples.

    :param y_pred: a x 1 vector, predicted labels of leaf nodes
    :param y_true: a x 1 vector, true labels of leaf nodes
    :return: float
    """

    # wrong = (y_pred[:len(y_true)] != y_true).sum()  # no longer necessary
    wrong = (y_pred != y_true).sum()
    error = wrong / len(y_true)
    return error


# tested
def assign_labels(L, u, v, T, num_samples):
    """
    Assigns label of root node to every leaf node in its subtree, using recursion.

    Todo - rewrite to avoid needing to specify u the first time

    :param L: V x 1 vector, where V is the number of nodes in T.
            Represents current label assignments all nodes in T.
    :param u: current node
    :param v: root node of subtree Tv
    :param T: Tree data structure
    :param num_samples: total number of samples in the data
    :return: V x 1 vector, representing updated label assignments all nodes in T
    """

    link = T[0]
    if u < num_samples:  # base case - u is a leaf node
        L[u] = L[v]
    else:
        # recursively search subtree rooted at left child of u to find leaves
        left_child = link[u - num_samples, 0]
        L = assign_labels(L, left_child, v, T, num_samples)

        # recursively search subtree rooted at right child of u to find leaves
        right_child = link[u - num_samples, 1]
        L = assign_labels(L, right_child, v, T, num_samples)

    return L


# tested
def get_leaves(leaves, v, T, num_samples):
    """
    Obtains indexes of all leaf nodes for the subtree Tv rooted at v, using recursion.

    :param leaves: list of indexes representing previously found leaves
    :param v: root node of subtree Tv
    :param T: Tree data structure
    :param num_samples: number of samples in the hierarchy
    :return: list of indexes representing leaves in the subtree Tv rooted at v, indexed relative to T.
            For example, leaves=[0,2] means nodes 0 and 2 within tree T are leaf nodes of Tv.
    """

    link = T[0]
    if v < num_samples:
        # base case, v is a leaf node
        leaves.append(v)
    else:
        # recursively search for leaves in left child of v
        left = link[v - num_samples, 0]
        leaves = get_leaves(leaves, left, T, num_samples)

        # recursively search for leaves in right child of v
        right = link[v - num_samples, 1]
        leaves = get_leaves(leaves, right, T, num_samples)

    return leaves


# ?? ask about what probabilities represent
# tested
def calculate_confidence_lower_bounds(n, pHat1):
    """
    Calculates lower confidence bounds for empirical labelings label=0 and label=1 for each node v in V,
    using Wald's approximation.

    :param n: V x 1 vector, where V is the number of nodes in T.
            n[u] is the number of points sampled from node u
    :param pHat1: V x 1 vector.
            pHat1[u] is the empirical probability of label=1 in tree node u
    :return: (V x 1 vector,V x 1 vector) Tuple representing (p0_LB, p1_LB) where
            p0_LB[u] is the lower bounds confidence for labeling all leaves in Tu with 0, and
            p1_LB[u] is the same for label=1.
    """
    # calculate delta label=1
    delta1 = (1 / n) + np.sqrt(pHat1 * (1 - pHat1) / n)  # V x 1 vector

    # label 0 lower bound
    p0_tmp = (1 - pHat1) - delta1
    p0_LB = np.fmax(p0_tmp, 0)

    # label 1 lower bound
    p1_tmp = pHat1 - delta1
    p1_LB = np.fmax(p1_tmp, 0)  # fmax is component-wise and ignores nan (unlike max)

    return p0_LB, p1_LB


# tested
def _identify_admissible_sets(p0_LB, p1_LB):
    """
    Identifies admissible set of labels for each node v in V, assuming that
    v contains some known labels.

    :param p0_LB: V x 1 vector
    :param p1_LB: V x 1 vector
    :return: (V x 1 vector, V x 1 vector) Tuple representing (A0,A1) where
            A0[v] is true if the confidence of label=0 for v is greater than 0.33, and
            A1[v] is the same for label=1.
    """
    p0_err_LB = 1 / 3  # with two possible labels, lower bound on the error rate simplifies to 1/3
    p1_err_LB = 1 / 3  # same
    A0 = p0_LB > p0_err_LB
    A1 = p1_LB > p1_err_LB
    return A0, A1


# tested
def _estimate_pruning_error(pHat1, A0, A1):
    """
    Calculates empirical estimate of the error of a pruning.

    :param pHat1: V x 1 vector
    :param A0: V x 1 vector
    :param A1: V x 1 vector
    :return: (V x 1 vector, V x 1 vector) Tuple representing (e0_tilde, e1_tilde) where
            e0_tilde[v] is the conservative error when all of subtree Tv is labeled with 0, and
            e1_tilde[v] is the same for label=1.
    """

    # error with label=1
    e1 = 1 - pHat1
    e1_tilde = np.ones(len(e1))
    e1_tilde[A1] = e1[A1]  # if label=1 for v is admissible, use e1 otherwise use 1

    # error with label=0
    e0 = pHat1
    e0_tilde = np.ones(len(e0))
    e0_tilde[A0] = e0[A0]  # if label=0 for v is admissible, use e0 otherwise use 1

    return e0_tilde, e1_tilde


# tested
def _update_parent_error(i, T, A0, A1, err0, err1, err_v):
    """

    Todo - determine if I can set score to 0.0 if np.isnan(). It seems like I should be able to do that.

    :param i: ith node in tree T
    :param T: Tree data structure
    :param A0: V x 1 vector
    :param A1: V x 1 vector
    :param err0: V x 1 vector
    :param err1: V x 1 vector
    :param err_v: scalar
    :return: None
    """
    # T components
    sizes_of_subtrees = T[1]
    parents = T[2]

    # action depends on parent of i
    parent = parents[i]
    if parent == 0:
        return  # we've reached the root node in Tv

    # update score for parents
    w_v = sizes_of_subtrees[i] / sizes_of_subtrees[parent]  # fraction of nodes in parent subtree which are in i
    if A0[i]:
        if np.isnan(err0[parent]):  # no score has been set for parent of i
            err0[parent] = w_v * err_v
        else:
            err0[parent] += w_v * err_v

    if A1[i]:
        if np.isnan(err1[parent]):  # no score has been set for parent of i
            err1[parent] = w_v * err_v
        else:
            err1[parent] += w_v * err_v


# ?? what is this doing
# tested
def _find_best_option(n, v, T, A0, A1, e0_tilde, e1_tilde):
    """
    Chooses the pruning strategy which minimizes the error associated with pruning P.

    :param n: V x 1 vector
    :param v: root node for Tv
    :param T: Tree data structure
    :param A0: V x 1 vector
    :param A1: V x 1 vector
    :param e0_tilde: V x 1 vector
    :param e1_tilde: V x 1 vector
    :return: integer, represents best of four options
    """
    # setup scaffold to fill in
    err = np.zeros(len(n))
    err0 = np.full_like(n, np.nan, dtype=float)
    err1 = np.full_like(n, np.nan, dtype=float)

    # sum errors for all subtrees including root
    for i in range(len(n)):
        # update error for subtree rooted at i
        _update_parent_error(i, T, A0, A1, err0, err1, err[i])

        # retain smallest error for subtree i
        possible_errs_tmp = [err0[i], err1[i], e0_tilde[i], e1_tilde[i]]
        err[i] = np.nanmin(possible_errs_tmp)

    # ?? seems unnecessary to do this again
    # retain smallest error for root
    possible_errs_tmp = [err0[-1], err1[-1], e0_tilde[-1], e1_tilde[-1]]
    err[-1] = np.nanmin(possible_errs_tmp)

    # choose strategy which minimizes error for tree Tv rooted at v
    possible_errs_tmp = [err0[v], err1[v], e0_tilde[v], e1_tilde[v]]
    best_option = np.nanargmin(possible_errs_tmp)

    return best_option


# tested
def _P_best_after_pruning(v, T, num_samples):
    """

    :param v: root node of tree Tv
    :param T: Tree data structure
    :param num_samples: Number of samples in the data
    :return: array representing best pruning
    """
    # T components
    link = T[0]

    if v < num_samples:  # v is a leaf node

        # best pruning occurs at root
        P_best = np.array([v])
    else:
        # get left child of v
        left = link[v - num_samples, 0]

        # get right child of v
        right = link[v - num_samples, 1]

        # best pruning occurs at children of root
        P_best = np.array([left, right])

    return P_best


# ?? understand how this works
# tested
def _get_P_best_and_L_best(v, T, num_samples, best_option):
    """

    :param v: root node of tree Tv
    :param T: Tree data structure
    :param num_samples:
    :param best_option: option which produces the min score for the tree Tv rooted at v
    :return:
    """

    if best_option == 0:  # score0 is min
        L_best = 0
        P_best = _P_best_after_pruning(v, T, num_samples)
    elif best_option == 1:  # score1 is min
        L_best = 1
        P_best = _P_best_after_pruning(v, T, num_samples)
    elif best_option == 2:  # e0_tilde is min
        L_best = 0
        P_best = np.array([v])  # prune at v
    elif best_option == 3:  # e1_tilde is min
        L_best = 1
        P_best = np.array([v])  # prune at v
    else:
        raise ValueError('best must be an integer between 0 and 3')

    return P_best, L_best


def best_pruning_and_labeling(n, pHat1, v, T, num_samples):
    """
    Finds best pruning and labeling.

    :param n: V x 1 vector
    :param pHat1: V x 1 vector
    :param v: root node of tree T_v
    :param T: Tree data structure
    :param num_samples: number of samples in the data
    :return: (array,int) Tuple representing (P_best, L_best) where
            P_best is the best new pruning, and
            L_best is the best labeling for all nodes in subtree Tv rooted at v
    """

    p0_LB, p1_LB = calculate_confidence_lower_bounds(n, pHat1)
    A0, A1 = _identify_admissible_sets(p0_LB, p1_LB)
    e0_tilde, e1_tilde = _estimate_pruning_error(pHat1, A0, A1)
    best_option = _find_best_option(n, v, T, A0, A1, e0_tilde, e1_tilde)
    P_best, L_best = _get_P_best_and_L_best(v, T, num_samples, best_option)

    return P_best, L_best


# ?? why can't z be zero? there is a leaf node 0
# tested
def update_empirical(n, pHat1, v, z, label_z, T):
    """
    Update empirical counts and probabilities for all nodes u on path between nodes z and v
    if every node u in path is assigned the given label.
    
    :param n: V x 1 vector
    :param pHat1: V x 1 vector
    :param v:  root node of subtree Tv
    :param z: leaf node in subtree Tv
    :param label_z: queried label for node z
    :param T: Tree data structure
    :return: (V x 1 vector,V x 1 vector) Tuple representing updated (n,pHat1)
    """

    parents = T[2]
    while z <= v and z != 0:  # for each node in path between z and v
        l1_prev = n[z] * pHat1[z]
        n[z] += 1
        pHat1[z] = (l1_prev + label_z) / n[z]
        z = np.array([parents[int(z)]])  # get parent
    return n, pHat1
