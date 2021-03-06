"""
Code provided to help implementation of DH.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import optim


# provided
def load_data(filename, seed, filter_class):
    """
    Loads data and computes linkage via hierarchical clustering.

    |
    Note 1: If hierarchical clustering is performed on 4 samples, the total number of nodes represented by Z is 5.
    This is because the final clustering is not recorded because it is redundant:

    |
    |  Given : 0  1  2  3
    |  Step 1 combine 2 and 3 to get node 4   // 3 nodes in hierarchy
    |  Step 2 combine 1 and 4 to get node 5   // 5 nodes in hierarchy
    |  Stop - sample 0 is not given a node because we know the final clustering is 0 and 5.
    Hierarchical clustering does not record combining 0 and 5 to get 6, so the overall tree is missing 2 nodes.


    :param filename: Path to the data file.
    :param seed: Seed used in RNG.
    :param filter_class: List of classes to keep in the dataset.
    :return: (np.ndarray,np.ndarray,np.ndarray,np.ndarray, tree) Tuple represents
            ( X_train, y_train, X_test, y_test, T) where T is a 3-element tree consisting of the following:
            T[0] = linkage matrix from hierarchical clustering.
            T[1] = An array denoting the size of each subtree rooted at node i, where i indexes the array.
                   i.e. The number of all leaves in subtree rooted at node i (w_i in the paper).
            T[2] = dict where keys are nodes and values are the node's parent.
    """
    # read data
    df = pd.read_csv(filename)

    # filter out samples which are not in filter_class
    mask = df.Label == 0
    for x in filter_class:
        mask = mask | (df.Label == x)
    df = df[mask]

    # extract DataFrame features
    X = df.iloc[:, :8].to_numpy()

    # extract DataFrame labels
    # encode labels
    y = df.Label.astype('category').cat.codes.to_numpy()

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # build data structure T
    # perform hierarchical clustering over training set
    Z = linkage(X_train, method='ward')

    # build (n-1) x 2 linkage matrix where n is the number of instances in the training set
    # row i contains ids of clusters combined at ith iteration
    link = Z[:, :2].astype(int)

    # build subtrees for each node in the hierarchy
    n_samples = len(X_train)
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

    return X_train, y_train, X_test, y_test, T


# I created
def get_classifier(choice, X=None, y=None, seed=None):
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


# provided
def compute_error(L, labels):
    """
    Compute the error

    :param L: labeling of leaf nodes
    :param labels: true labels of each node

    :return: error of predictions
    """

    wrong = 0
    wrong = (L[:len(labels)] != labels).sum()
    error = wrong / len(labels)
    return error


def assign_labels(L, u, v, T, n_samples):
    """Assign labels to every leaf according to the label of the subtree's root node

    :param L: array of predicted labels for each node
    :param u: current node
    :param v: subtree's root node
    :param T: tree- 3 element list, see dh.py for description
    :param nsample: number of samples, 1000

    :returns L: array of predicted label for each node
    """

    link = T[0]
    if u < n_samples:
        L[u] = L[v]
        return L
    else:
        left = link[u - n_samples, 0]
        L = assign_labels(L, left, v, T, n_samples)
        right = link[u - n_samples, 1]
        L = assign_labels(L, right, v, T, n_samples)
        return L


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


def get_leaves(leaves, v, T, n_samples):
    """Get all leaf nodes in the subtree Tv rooted at v

    :param leaves: previously found leaves
    :param v: current root node
    :param T: Tree- 3 element list, see dh.py for description
    :param n_samples: number of samples

    :returns leaves: leaves in the subtree Tv rooted at v"""

    link = T[0]
    if v < n_samples:
        leaves.append(v)
        return leaves
    else:
        left = link[v - n_samples, 0]
        leaves = get_leaves(leaves, left, T, n_samples)
        right = link[v - n_samples, 1]
        leaves = get_leaves(leaves, right, T, n_samples)
        return leaves


def update_empirical(n, p1, v, z, l, T):
    ''' Update empirical counts and probabilities for all nodes u on path

    :param n: array with number of points sampled in the subtree rooted at each node
    :param p1: array with fraction of label = 1 in the subtree rooted at each node
    :param v: selected subtree Tv's root node
    :param z: selected leaf node in subtree Tv
    :param l: queried label for node z
    :param T: tree- 3 element list, see dh.py for details

    :returns n: array with updated number of points sampled in the subtree rooted at each node
    :returns p1: array with updated fraction of label = 1 in the subtree rooted at each node
    '''

    parents = T[2]
    while z <= v and z != 0:
        l1 = n[z] * p1[z]
        n[z] += 1
        p1[z] = (l1 + l) / n[z]
        z = np.array([parents[int(z)]])
    return n, p1
