import numpy as np
import packages.zlg.zlg as zlg


def test__calculate_weights_2():

    X = np.array([[1, 2],
                  [3, 4]])

    # calculate expected
    a = -1 / np.std(X)
    inner = np.multiply(np.array([[0.0,8.0],[8.0,0.0]]),a )
    expected = np.exp(inner)

    actual = zlg._calculate_weights_2(X)
    np.testing.assert_allclose(actual,expected, atol=1e-16)  # rounding errors cause problems with exact comparison


def test__calculate_weights():
    X = np.array([[1, 2],
                  [3, 5]])

    expected = np.array([[1, np.exp(-8.0)],
                         [np.exp(-8.0), 1]])
    actual = zlg._calculate_weights(X)
    np.testing.assert_array_equal(actual, expected)


def test__construct_weight_matrix():
    weights = np.array([[1, 2],
                        [3, 4]])
    t = 3.0
    expected = np.array([[0.0, 0.0],
                         [3, 4]])
    actual = zlg._construct_weight_matrix(weights, t)
    np.testing.assert_array_equal(actual, expected)


def test__construct_diagonal_matrix():
    weights = np.array([[1, 2],
                        [3, 4]])
    expected = np.array([[3, 0],
                         [0, 7]])
    actual = zlg._construct_diagonal_matrix(weights)
    np.testing.assert_array_equal(actual, expected)


def test__subtract_matrices():
    W = np.array([[1, 2],
                  [3, 4]])
    D = np.array([[3, 0],
                  [0, 7]])
    expected = np.array([[2, -2],
                         [-3, 3]])
    actual = zlg._subtract_matrices(D, W)
    np.testing.assert_array_equal(actual, expected)


def test_laplacian_matrix():
    t = 0.0
    X = np.array([[1, 2],
                  [3, 5]])
    expected = np.array([[np.exp(-8.0), -np.exp(-8.0)],
                         [-np.exp(-8.0), np.exp(-8.0)]])
    actual = zlg.laplacian_matrix(X, t)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


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


def test__rearrange_laplacian_matrix():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 3]  # label instance 1 and 4
    unlabeled = [1, 2]  # don't label instance 2 and 3

    expected = np.array([[1, 4, 2, 3],
                         [13, 16, 14, 15],
                         [5, 8, 6, 7],
                         [9, 12, 10, 11]])
    actual = zlg._rearrange_laplacian_matrix(L, labeled, unlabeled)
    np.testing.assert_array_equal(actual, expected)


def test_minimum_energy_solution():
    L = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    labeled = [0, 3]  # label instance 1 and 4
    unlabeled = [1, 2]  # don't label instance 2 and 3
    f_l = np.array([[1], [0]])

    expected_f_u = np.array([[-2],
                             [1]])

    expected_uu_inv = np.array([[-2.75, 1.75],
                                [2.5, -1.5]])
    actual_f_u, actual_uu_inv = zlg.minimum_energy_solution(L, labeled, unlabeled, f_l)
    np.testing.assert_allclose(actual_f_u, expected_f_u,
                               atol=1e-16)  # rounding errors cause problems with exact comparison
    np.testing.assert_allclose(expected_uu_inv, actual_uu_inv, atol=1e-16)


def test__add_point_to_f_u():
    f_u = np.array([-2, 1])

    uu_inv = np.array([[-2, 1],
                       [2, -1]])
    k = 0
    y_k = 1

    expected = np.array([1, -2])
    actual = zlg._add_point_to_f_u(f_u, uu_inv, k, y_k)
    np.testing.assert_array_equal(actual, expected)


def test__expected_risk():
    f_u = np.array([-2, 1])

    expected = -2.0
    actual = zlg._expected_risk(f_u)
    assert actual == expected


def test_expected_estimated_risk():
    f_u = np.array([-2, 1])

    uu_inv = np.array([[-2, 1],
                       [2, -1]])
    k = 0

    # helpful info
    # f_u_plus_xk0 = np.array([[0],
    #                         [-1]])
    # Rhat_f_plus_xk0 = -2.0
    # f_u_plus_xk1 = np.array([[1],
    #                         [-2]])
    # Rhat_f_plus_xk1 = -1.0

    expected = 1.0
    actual = zlg.expected_estimated_risk(f_u, uu_inv, k)
    assert actual == expected


def test_zlg_query():
    f_u = np.array([-2, 1])

    uu_inv = np.array([[-2, 1],
                       [2, -1]])
    num_labeled = 2
    num_samples = 4

    expected = 1
    actual = zlg.zlg_query(f_u, uu_inv, num_labeled, num_samples)
    assert actual == expected
