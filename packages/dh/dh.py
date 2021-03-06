import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.model_selection import train_test_split


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
