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
