import packages.dh.helper as helper
import numpy as np


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