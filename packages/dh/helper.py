"""
Code provided to help implementation of DH.
"""

import numpy as np

from scipy.cluster.hierarchy import linkage

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim


# I created
# tested
def generate_T(X):
    """
    Builds data structure T to contain linkage, subtree sizes, and subtrees.

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

    :param X: n x m dataset
    :return: List representing T
    """

    # perform hierarchical clustering over training set
    Z = linkage(X, method='ward')

    # build (n-1) x 2 linkage matrix where n is the number of instances in the training set
    # row i contains ids of clusters combined at ith iteration
    link = Z[:, :2].astype(int)

    # build subtrees for each node in the hierarchy
    n_samples = len(X)
    n_nodes = link[-1, -1]  # the number of nodes generated via clustering - See Note 1
    n_subtree = n_nodes + 2  # add two additional nodes to account for final clustering - See Note 1
    subtree_sizes = np.zeros(n_subtree)  # scaffold to list number of leaves per subtree
    subtree_sizes[:n_samples] = 1  # subtrees containing an individual sample before clustering have a single leaf

    # link subtrees together and store relationship in dictionary
    # ?? shouldn't root be set to -1 or something not representing another node in the tree - we have a node 0
    parents = {2 * (n_samples - 1): 0}  # dictionary relating subtrees together, with root of tree set to 0
    for i in range(len(link)):
        left_child = link[i, 0]
        right_child = link[i, 1]
        parent = i + n_samples
        parents[left_child] = parent
        parents[right_child] = parent

        # calculates number of leaves in grouped subtree formed at each level in the hierarchy
        subtree_sizes[parent] = subtree_sizes[left_child] + subtree_sizes[right_child]

    # build data structure to access all components
    T = [link, subtree_sizes, parents]

    return T


# I created
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
    Compute the error

    :param y_pred: labeling of leaf nodes
    :param y_true: true labels of each node

    :return: error of predictions
    """

    wrong = (y_pred[:len(y_true)] != y_true).sum()
    error = wrong / len(y_true)
    return error


# tested
def assign_labels(y_pred, u, root, T, n_samples):
    """Assigns label of root node to every leaf in its subtree.

    :param y_pred: array of predicted labels for each node
    :param u: current node
    :param root: subtree's root node
    :param T: data structure representing hierarchy
    :param n_samples: number of samples in the hierarchy

    :return: array of predicted labels for each node
    """

    link = T[0]
    if u < n_samples:  # base case - u is a leaf node
        y_pred[u] = y_pred[root]
    else:
        # recursive search subtree rooted at left child of u to find leaves
        left_child = link[u - n_samples, 0]
        y_pred = assign_labels(y_pred, left_child, root, T, n_samples)

        # recursive search subtree rooted at right child of u to find leaves
        right_child = link[u - n_samples, 1]
        y_pred = assign_labels(y_pred, right_child, root, T, n_samples)

    return y_pred


def best_pruning_and_labeling(n, p1, v, T, n_samples):
    """ update admissible A and compute scores; find best pruning and labeling

    :param n: array with number of points sampled in the subtree rooted at each node
    :param p1: array with fraction of label = 1 in the subtree rooted at each node
    :param v: selected subtree Tv's root node
    :param T: tree- 3 element list, see dh.py for description
    :param n_samples: number of samples, 1000

    :returns P_best: the best new pruning
    :returns L_best: the best labeling for v
    """
    p0_tmp = 1 - p1 - ((1 / n) + np.sqrt(p1 * (1 - p1) / n))
    p0_LB = np.fmax(p0_tmp, 0)  # fmax is componentwise and ignores nans, unlike max

    p1_tmp = p1 - ((1 / n) + np.sqrt(p1 * (1 - p1) / n))
    p1_LB = np.fmax(p1_tmp, 0)

    A0 = p0_LB > 1 / 3
    A1 = p1_LB > 1 / 3
    e1 = 1 - p1
    e1_tilde = np.ones(len(e1))
    e1_tilde[A1] = e1[A1]
    e0 = p1
    e0_tilde = np.ones(len(e0))
    e0_tilde[A0] = e1[A0]

    score0 = np.full_like(n, np.nan)
    score1 = np.full_like(n, np.nan)
    score = np.zeros(len(n))

    link = T[0]
    sizes_of_subtrees = T[1]
    parents = T[2]

    for i in range(len(n)):
        parent = parents[i]
        if parent == 0:
            break
        if A0[i]:
            if np.isnan(score0[parent]):
                score0[parent] = sizes_of_subtrees[i] / sizes_of_subtrees[parent] * score[i]
            else:
                score0[parent] += sizes_of_subtrees[i] / sizes_of_subtrees[parent] * score[i]
        if A1[i]:
            if np.isnan(score1[parent]):
                score1[parent] = sizes_of_subtrees[i] / sizes_of_subtrees[parent] * score[i]
            else:
                score1[parent] += sizes_of_subtrees[i] / sizes_of_subtrees[parent] * score[i]
        possible_scores_tmp = [score0[i], score1[i], e0_tilde[i], e1_tilde[i]]
        score[i] = np.nanmin(possible_scores_tmp)

    possible_scores_tmp = [score0[-1], score1[-1], e0_tilde[-1], e1_tilde[-1]]
    score[-1] = np.nanmin(possible_scores_tmp)

    scores_tmp = [score0[v], score1[v], e0_tilde[v], e1_tilde[v]]
    best = np.nanargmin(scores_tmp)

    if best == 0:
        L_best = 0
        if v < n_samples:
            P_best = np.array([v])
        else:
            left = link[v - n_samples, 0]
            right = link[v - n_samples, 1]
            P_best = np.array([left, right])
    elif best == 1:
        L_best = 1
        if v < n_samples:
            P_best = np.array([v])
        else:
            left = link[v - n_samples, 0]
            right = link[v - n_samples, 1]
            P_best = np.array([left, right])
    elif best == 2:
        L_best = 0
        P_best = np.array([v])
    else:
        L_best = 1
        P_best = np.array([v])

    return P_best, L_best


# tested
def get_leaves(leaves, v, T, n_samples):
    """Get all leaf nodes in the subtree T_v rooted at v, recursively.

    :param leaves: previously found leaves. Supply empty list if this is the first iteration.
    :param v: current root node
    :param T: data structure representing hierarchy
    :param n_samples: number of samples in the hierarchy

    :return: indexes of leaves in the subtree T_v rooted at v
    """

    link = T[0]
    if v < n_samples:  # base case - v is a leaf node
        leaves.append(v)
    else:
        # recursively add leaves from left child of v
        left = link[v - n_samples, 0]
        leaves = get_leaves(leaves, left, T, n_samples)

        # recursively add leaves from right child of v
        right = link[v - n_samples, 1]
        leaves = get_leaves(leaves, right, T, n_samples)

    return leaves


# ?? why can't z be zero? there is a leaf node 0
# tested
def update_empirical(n, p1, v, z, label_z, T):
    """
    Update empirical counts and probabilities for all nodes u on path between nodes z and v
    if every node u in path is assigned the given label.
    
    :param n: array with number of points sampled in the subtree rooted at each node
    :param p1: array with fraction of label = 1 in the subtree rooted at each node 
    :param v:  root node of selected subtree T_v
    :param z: selected leaf node in subtree T_vv
    :param label_z: queried label for node z
    :param T: data structure representing hierarchy
    :return: (array,array) Tuple representing (n,p1) where
            n is an array with updated number of points sampled in the subtree rooted at each node;
            p1 is an array with updated fraction of label = 1 in the subtree rooted at each node
    """

    parents = T[2]
    while z <= v and z != 0:  # for each node in path between z and v
        l1_prev = n[z] * p1[z]
        n[z] += 1
        p1[z] = (l1_prev + label_z) / n[z]
        z = np.array([parents[int(z)]])  # get parent
    return n, p1