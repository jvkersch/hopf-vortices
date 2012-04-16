import numpy as np

from vectors import vectorize


def cayley(xi):
    """Applies Cayley map to the Lie algebra element xi.

    INPUT: 

      -- ``xi`` - Lie algebra matrix.

    OUTPUT:

    Group element.

    """

    if xi.ndim == 2:
        xi = np.array([xi])

    xi = np.array(xi)
    m, _, n = xi.shape
    I = np.eye(m)
    M = np.empty((m, m, n), dtype=complex)

    for k in xrange(0, n):
        element = xi[:, :, k]
        M[:, :, k] = np.dot(element + I, np.linalg.inv(element - I))

    return M
