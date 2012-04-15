import numpy as np

def cayley(xi):
    """Applies Cayley map to the Lie algebra element xi.

    INPUT: 

      -- ``xi`` - Lie algebra matrix.

    OUTPUT:

    Group element.

    """
    m, n = xi.shape[1]
    if m != n:
        raise ValueError, "Lie algebra element must be square matrix."
    I = np.eye(m)
    return np.dot(xi + I, np.linalg.inv(xi - I))
