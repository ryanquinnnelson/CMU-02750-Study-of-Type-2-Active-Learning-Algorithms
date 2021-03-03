import numpy as np
import packages.zlg.zlg as zlg


# def test__calculate_weights():
#
#     X = np.array([[1, 2], [3, 4]])
#
#     # calculate expected
#     inner = np.array([[0.0,-2.0],[-2.0,0.0]])
#     expected = np.exp(inner)
#
#     actual = zlg._calculate_weights(X)
#     np.testing.assert_allclose(actual,expected, atol=1e-16)  # rounding errors cause problems with exact comparison


def test__construct_weight_matrix():
    weights = np.array([[1, 2], [3, 4]])
    t = 3.0
    expected = np.array([[0.0, 0.0], [3, 4]])
    actual = zlg._construct_weight_matrix(weights, t)
    np.testing.assert_array_equal(actual, expected)


def test__construct_diagonal_matrix():
    weights = np.array([[1, 2], [3, 4]])
    expected = np.array([[3, 0], [0, 7]])
    actual = zlg._construct_diagonal_matrix(weights)
    np.testing.assert_array_equal(actual, expected)


def test__construct_laplacian_matrix():
    W = np.array([[1, 2], [3, 4]])
    D = np.array([[3, 0], [0, 7]])
    expected = np.array([[2, -2], [-3, 3]])
    actual = zlg._construct_laplacian_matrix(D, W)
    np.testing.assert_array_equal(actual, expected)


def test__construct_ll_two_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 3]  # select instance 1 and 4
    expected = np.array([[1, 4],
                         [13, 16]])

    actual = zlg._construct_ll(L, labeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_ll_three_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [1, 2, 3]  # select instance 2,3,4
    expected = np.array([[6, 7, 8],
                         [10, 11, 12],
                         [14, 15, 16]])

    actual = zlg._construct_ll(L, labeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_ll_one_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0]  # select instance 1
    expected = np.array([[1]])

    actual = zlg._construct_ll(L, labeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_uu_two_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    unlabeled = [0, 3]  # select instance 1 and 4
    expected = np.array([[1, 4],
                         [13, 16]])

    actual = zlg._construct_uu(L, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_uu_three_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    unlabeled = [1, 2, 3]  # select instance 2,3,4
    expected = np.array([[6, 7, 8],
                         [10, 11, 12],
                         [14, 15, 16]])

    actual = zlg._construct_uu(L, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_uu_one_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    unlabeled = [0]  # select instance 1
    expected = np.array([[1]])

    actual = zlg._construct_uu(L, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_lu_two_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 3]  # label instance 1 and 4
    unlabeled = [1, 2]  # don't label instance 2 and 3
    expected = np.array([[2, 3],
                         [14, 15]])

    actual = zlg._construct_lu(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_lu_three_selected_together():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [1, 2, 3]  # select instance 2,3,4
    unlabeled = [0]
    expected = np.array([[5],
                         [9],
                         [13]])

    actual = zlg._construct_lu(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_lu_one_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0]  # select instance 1
    unlabeled = [1, 2, 3]
    expected = np.array([[2, 3, 4]])

    actual = zlg._construct_lu(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_lu_three_selected_sep():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 2, 3]  # select instance 1
    unlabeled = [1]
    expected = np.array([[2],
                         [10],
                         [14]])

    actual = zlg._construct_lu(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_ul_two_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 3]  # label instance 1 and 4
    unlabeled = [1, 2]  # don't label instance 2 and 3
    expected = np.array([[5, 8],
                         [9, 12]])

    actual = zlg._construct_ul(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_ul_three_selected_together():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 1, 2]  # select instance 2,3,4
    unlabeled = [3]
    expected = np.array([[13, 14, 15]])

    actual = zlg._construct_ul(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_ul_one_selected():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0]  # select instance 1
    unlabeled = [1, 2, 3]
    expected = np.array([[5], [9], [13]])

    actual = zlg._construct_ul(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test__construct_ul_three_selected_sep():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 2, 3]  # select instance 1
    unlabeled = [1]
    expected = np.array([[5, 7, 8]])

    actual = zlg._construct_ul(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)
