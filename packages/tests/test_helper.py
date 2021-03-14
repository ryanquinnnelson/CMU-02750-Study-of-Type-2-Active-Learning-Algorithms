import packages.dh.helper as helper
import numpy as np
import pytest


def test_generate_T():
    """
    Hierarchy for reference:
             |
        _____6_____
       |           |
     __4__       __5__
    |     |     |     |
    0     3     1     2


    :return:
    """
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    expected_t0 = np.array([[0, 3],
                            [1, 2],
                            [4, 5]])
    expected_t1 = np.array([1., 1., 1., 1., 2., 2., 4.])
    expected_t2 = {6: 0, 0: 4, 3: 4, 1: 5, 2: 5, 4: 6, 5: 6}

    actual = helper.generate_T(X_train)
    np.testing.assert_array_equal(actual[0], expected_t0)
    np.testing.assert_array_equal(actual[1], expected_t1)
    assert actual[2] == expected_t2


def test_compute_error():
    y_pred = np.array([1, 0, 0, 1])
    y_true = np.array([1, 0, 0, 0])
    expected = 0.25
    actual = helper.compute_error(y_pred, y_true)
    assert actual == expected


def test_assign_labels_u_less_than_n_samples():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)

    y_pred = np.array([1, 0, 0, 0, 1, 0, 1])
    u = 1
    root = 1
    n_samples = 4

    expected = np.array([1, 0, 0, 0, 1, 0, 1])
    actual = helper.assign_labels(y_pred, u, root, T, n_samples)
    np.testing.assert_array_equal(actual, expected)


def test_assign_labels_start_root():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)

    y_pred = np.array([0, 0, 0, 0, 0, 0, 1])
    u = 6
    v = 6
    n_samples = 4

    expected = np.array([1, 1, 1, 1, 0, 0, 1])
    actual = helper.assign_labels(y_pred, u, v, T, n_samples)
    np.testing.assert_array_equal(actual, expected)


def test_assign_labels_start_middle():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)

    y_pred = np.array([0, 0, 0, 0, 0, 0, 1])
    u = 4
    v = 6
    n_samples = 4

    expected = np.array([1, 0, 0, 1, 0, 0, 1])
    actual = helper.assign_labels(y_pred, u, v, T, n_samples)
    np.testing.assert_array_equal(actual, expected)


def test_get_leaves_middle():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    leaves = []
    n_samples = 4
    v = 4
    expected = [0, 3]
    actual = helper.get_leaves(leaves, v, T, n_samples)
    assert actual == expected


def test_get_leaves_root():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    leaves = []
    n_samples = 4
    v = 6
    expected = [0, 3, 1, 2]
    actual = helper.get_leaves(leaves, v, T, n_samples)
    assert actual == expected


def test_update_empirical_z_greater_than_v():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    n = np.array([1, 1, 1, 1, 1, 1, 1])
    p1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    v = 1
    z = 2
    label_z = 0
    expected = (n, p1)
    actual = helper.update_empirical(n, p1, v, z, label_z, T)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_update_empirical_z_equals_zero():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    n = np.array([1, 1, 1, 1, 1, 1, 1])
    p1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    v = 1
    z = 0
    label_z = 0
    expected = (n, p1)
    actual = helper.update_empirical(n, p1, v, z, label_z, T)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_update_empirical_one_step():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    n_prev = np.array([1, 2, 3, 4, 5, 6, 7])
    p1_prev = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    v = 4
    z = 4
    label_z = 0

    # calc expected
    n = np.array([1, 2, 3, 4, 6, 6, 7])
    p1 = np.array([0.15, 0.25, 0.35, 0.45, 2.75 / 6, 0.65, 0.75])
    expected = (n, p1)

    actual = helper.update_empirical(n_prev, p1_prev, v, z, label_z, T)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_update_empirical_multiple_steps():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    n_prev = np.array([1, 2, 3, 4, 5, 6, 7])
    p1_prev = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    v = 4
    z = 3
    label_z = 0

    # calc expected
    n = np.array([1, 2, 3, 5, 6, 6, 7])
    p1 = np.array([0.15, 0.25, 0.35, 0.36, 2.75 / 6, 0.65, 0.75])
    expected = (n, p1)

    actual = helper.update_empirical(n_prev, p1_prev, v, z, label_z, T)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_calculate_confidence_lower_bounds_test_one():
    n = np.array([100, 100, 100, 100])
    p1 = np.array([0.9, 0.8, 0.4, 0.6])

    wald_term = np.array([0.01, 0.01, 0.01, 0.01]) + np.sqrt(p1 * (1 - p1) / n)
    p0_LB = (1 - p1) - wald_term
    p1_LB = p1 - wald_term

    expected = p0_LB, p1_LB
    actual = helper.calculate_confidence_lower_bounds(n, p1)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_calculate_confidence_lower_bounds_test_two():
    n = np.array([100, 100, 100, 100, 200, 200, 400])  # number of nodes
    p1 = np.array([0.9, 0.4, 0.8, 0.5, 0.9, 0.5, 0.7])

    p0_LB = np.array([0.06, 0.54101021, 0.15, 0.44, 0.0737868,
                      0.45964466, 0.27458712])
    p1_LB = np.array([0.86, 0.34101021, 0.75, 0.44, 0.8737868,
                      0.45964466, 0.67458712])

    expected = p0_LB, p1_LB
    actual = helper.calculate_confidence_lower_bounds(n, p1)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-08)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-08)


def test__identify_admissible_sets():
    p0 = np.array([0.1, 0.8, 0.2, 0.6])
    p1 = 1 - p0

    expected = np.array([False, True, False, True]), np.array([True, False, True, True])
    actual = helper._identify_admissible_sets(p0, p1)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test__estimate_pruning_error():
    p1 = np.array([0.9, 0.2, 0.8, 0.4])
    A0 = np.array([True, False, True, True])
    A1 = np.array([False, True, False, True])

    expected = np.array([0.9, 1, 0.8, 0.4]), np.array([1, 0.8, 1, 0.6])
    actual = helper._estimate_pruning_error(p1, A0, A1)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-16)  # rounding error
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-16)


def test__update_parent_error_root_node():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    i = 6
    n = np.array([1, 1, 1, 1, 1, 1, 1])
    A0 = np.array([True, False, True, True, True, True, False])
    A1 = np.array([False, True, False, True, False, False, False])
    score0 = np.full_like(n, np.nan, dtype=float)
    score1 = np.full_like(n, np.nan, dtype=float)
    i_score = 0.0

    helper._update_parent_error(i, T, A0, A1, score0, score1, i_score)
    np.testing.assert_array_equal(score0, np.full_like(n, np.nan, dtype=float))
    np.testing.assert_array_equal(score1, np.full_like(n, np.nan, dtype=float))


def test__update_parent_error_leaf_node_parent_nan():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    i = 0
    n = np.array([1, 1, 1, 1, 1, 1, 1])
    A0 = np.array([True, False, True, True, True, True, False])
    A1 = np.array([False, True, False, True, False, False, False])
    score0 = np.full_like(n, np.nan, dtype=float)
    score1 = np.full_like(n, np.nan, dtype=float)
    i_score = 0.0

    # generate expected
    expected_score0 = np.full_like(n, np.nan, dtype=float)
    expected_score0[4] = 0.0

    helper._update_parent_error(i, T, A0, A1, score0, score1, i_score)

    np.testing.assert_array_equal(score0, expected_score0)
    np.testing.assert_array_equal(score1, np.full_like(n, np.nan, dtype=float))


def test__find_best_option():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 6
    n = np.array([1, 1, 1, 1, 2, 2, 4])
    A0 = np.array([True, False, True, True, True, True])
    A1 = np.array([False, True, False, True, False, False])
    e0_tilde = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    e1_tilde = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])

    expected = 0
    actual = helper._find_best_option(n, v, T, A0, A1, e0_tilde, e1_tilde)
    assert actual == expected


def test__P_best_after_pruning_v_less_than_n_samples():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 1
    n_samples = 4

    expected = np.array([v])
    actual = helper._P_best_after_pruning(v, T, n_samples)
    np.testing.assert_array_equal(actual, expected)


def test__P_best_after_pruning_v_greater_than_n_samples():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 6
    n_samples = 4

    expected = np.array([4, 5])
    actual = helper._P_best_after_pruning(v, T, n_samples)
    np.testing.assert_array_equal(actual, expected)


def test__get_P_best_and_L_best_for_best_equals_0():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 6
    n_samples = 4
    best = 0
    L_expected = 0
    P_expected = np.array([4, 5])
    P_actual, L_actual = helper._get_P_best_and_L_best(v, T, n_samples, best)
    assert L_actual == L_expected
    np.testing.assert_array_equal(P_actual, P_expected)


def test__get_P_best_and_L_best_for_best_equals_1():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 6
    n_samples = 4
    best = 1
    L_expected = 1
    P_expected = np.array([4, 5])
    P_actual, L_actual = helper._get_P_best_and_L_best(v, T, n_samples, best)
    assert L_actual == L_expected
    np.testing.assert_array_equal(P_actual, P_expected)


def test__get_P_best_and_L_best_for_best_equals_2():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 6
    n_samples = 4
    best = 2
    L_expected = 0
    P_expected = np.array([v])
    P_actual, L_actual = helper._get_P_best_and_L_best(v, T, n_samples, best)
    assert L_actual == L_expected
    np.testing.assert_array_equal(P_actual, P_expected)


def test__get_P_best_and_L_best_for_best_equals_3():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 6
    n_samples = 4
    best = 3
    L_expected = 1
    P_expected = np.array([v])
    P_actual, L_actual = helper._get_P_best_and_L_best(v, T, n_samples, best)
    assert L_actual == L_expected
    np.testing.assert_array_equal(P_actual, P_expected)


def test__get_P_best_and_L_best_raises_error():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    v = 6
    n_samples = 4
    best = 4
    L_expected = 0
    P_expected = np.array([4, 5])

    with pytest.raises(ValueError):
        helper._get_P_best_and_L_best(v, T, n_samples, best)
