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


def get_classifier(X, y, choice, seed=None):
    model = None

    if choice == 'Logistic Regression':
        lr = LogisticRegression()
        model = lr.fit(X, y)
    elif choice == 'Random Forest':
        N_estimator_rf = 20
        MAX_depth_rf = 6
        rf = RandomForestClassifier(n_estimators=N_estimator_rf,
                                    max_depth=MAX_depth_rf, random_state=seed)
        model = rf.fit(X, y)
    elif choice == 'Gradient Boosting Decision Tree':
        N_estimator_gbdt = 20
        gbdt_max_depth = 6
        gbdt = GradientBoostingClassifier(n_estimators=N_estimator_gbdt,
                                          learning_rate=0.1,
                                          max_depth=gbdt_max_depth,
                                          random_state=seed)
        model = gbdt.fit(X, y)
    elif choice == 'Neural Net':

        # 3-Layer fully connected NN
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
                        print("[%d|%d] epoch loss:%.2f" % (epoch_i + 1, epoches, loss_record / report_epoch))
                        loss_record = 0.0
                    if epoch_i >= epoches:
                        break

            def score(self, X_test, y_test):
                X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
                y_pred_test = self.model(X_test_tensor)
                y_output = torch.argmax(y_pred_test, axis=1).numpy()
                return (y_output == y_test).mean()

        nn = NNClassifier(feature_n=X.shape[1], class_n=len(np.unique(y)))
        model = nn.fit(X, y)

    return model
