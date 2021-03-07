import numpy as np
import packages.dh.helper as helper
import packages.dh.dh as dh


def test__get_proportional_weights_one_node():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    num_samples = 4
    P = np.array([6])
    expected = np.array([1.0])
    actual = dh._get_proportional_weights(P, T, num_samples)
    np.testing.assert_array_equal(actual, expected)


def test__get_proportional_weights_two_node():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    num_samples = 4
    P = np.array([4, 5])
    expected = np.array([0.5, 0.5])
    actual = dh._get_proportional_weights(P, T, num_samples)
    np.testing.assert_array_equal(actual, expected)


def test__proportional_selection_root():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)

    P = np.array([6])
    num_samples = 4
    expected = 6
    actual = dh._proportional_selection(P, T, num_samples)
    assert actual == expected


def test__proportional_selection_two():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)

    P = np.array([4, 5])
    num_samples = 4
    num_5 = 0
    num_6 = 0
    total = 10000
    for i in range(total):
        actual = dh._proportional_selection(P, T, num_samples)
        if actual == 5:
            num_5 += 1
        else:
            num_6 += 1
    assert 0.49 <= round(num_5 / total, 2) <= 0.51
