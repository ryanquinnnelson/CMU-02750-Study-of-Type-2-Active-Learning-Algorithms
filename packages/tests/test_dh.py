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


def test__get_proportional_weights_three_nodes_order_1():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    num_samples = 4
    P = np.array([5, 3, 0])
    expected = np.array([0.5, 0.25, 0.25])
    actual = dh._get_proportional_weights(P, T, num_samples)
    np.testing.assert_array_equal(actual, expected)


def test__get_proportional_weights_three_nodes_order_2():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    num_samples = 4
    P = np.array([3, 5, 0])
    expected = np.array([0.25, 0.5, 0.25])
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
    num_4 = 0
    total = 10000
    for i in range(total):
        actual = dh._proportional_selection(P, T, num_samples)
        if actual == 5:
            num_5 += 1
        else:
            num_4 += 1
    assert 0.49 <= round(num_5 / total, 2) <= 0.51


def test__proportional_selection_three():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)

    P = np.array([3, 5, 0])
    num_samples = 4
    num_5 = 0
    num_0 = 0
    num_3 = 0
    total = 10000
    for i in range(total):
        actual = dh._proportional_selection(P, T, num_samples)
        if actual == 5:
            num_5 += 1
        elif actual == 0:
            num_0 += 1
        else:
            num_3 += 1
    assert 0.49 <= round(num_5 / total, 2) <= 0.51
    assert 0.24 <= round(num_0 / total, 2) <= 0.26
    assert 0.24 <= round(num_3 / total, 2) <= 0.26


def test__get_confidence_adjusted_weights_one_node():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    num_samples = 4
    P = np.array([6])

    n = np.array([100, 100, 100, 100, 200, 200, 400])  # number of nodes
    pHat1 = np.array([0.9, 0.4, 0.8, 0.5, 0.9, 0.5, 0.7])

    expected = np.array([1.0])
    actual = dh._get_confidence_adjusted_weights(P, T, num_samples, n, pHat1)
    np.testing.assert_array_equal(actual, expected)


def test__get_confidence_adjusted_weights_two_nodes():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    num_samples = 4
    P = np.array([4, 5])

    n = np.array([100, 100, 100, 100, 200, 200, 400])  # number of nodes
    pHat1 = np.array([0.9, 0.4, 0.8, 0.5, 0.9, 0.5, 0.7])

    # calculate expected
    p1_LB = np.array([0.86, 0.34101021, 0.75, 0.44, 0.8737868,
                      0.45964466, 0.67458712])

    p1_LB_P = p1_LB[P]
    wv = np.array([0.5, 0.5])

    p = wv * (1 - p1_LB_P)
    expected = p / sum(p)
    actual = dh._get_confidence_adjusted_weights(P, T, num_samples, n, pHat1)
    np.testing.assert_allclose(actual, expected, atol=1e-08)


def test__confidence_adjusted_selection():
    X_train = np.array([[0.41, 0.59, 0.65, 0.14, 0.5, 0., 0.49, 0.33],
                        [0.55, 0.53, 0.54, 0.4, 0.5, 0., 0.48, 0.22],
                        [0.38, 0.38, 0.54, 0.24, 0.5, 0., 0.54, 0.22],
                        [0.49, 0.51, 0.52, 0.13, 0.5, 0., 0.51, 0.33]])

    T = helper.generate_T(X_train)
    num_samples = 4
    P = np.array([4, 5])

    n = np.array([100, 100, 100, 100, 200, 200, 400])  # number of nodes
    pHat1 = np.array([0.9, 0.4, 0.8, 0.5, 0.9, 0.5, 0.7])

    num_5 = 0
    num_4 = 0
    total = 10000
    for i in range(total):
        actual = dh._confidence_adjusted_selection(P, T, num_samples, n, pHat1)
        if actual == 5:
            num_5 += 1
        else:
            num_4 += 1

    assert 0.80 <= round(num_5 / total, 2) <= 0.82
