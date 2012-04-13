import numpy as np

def hstack(v, n):
    """Return array with vector v in each of the n columns"""
    return np.kron(np.ones((n, 1)), v).T

def normalize(array):
    """Normalize rows of array to unit length. Array can be real or complex."""

    norms = np.sqrt(np.sum(array*array.conjugate(), axis=1))
    return array/hstack(norms, array.shape[1])


def hopf(psi):
    """Applies Hopf fibration row-wise.

    INPUT:

      -- ``psi`` - complex Nx2 array with unit vectors on three-sphere.

    OUTPUT:

      Real-valued 3xN array of corresponding 3-vectors on two-sphere.

    """

    Z = psi[:, 0]
    U = psi[:, 1]

    XY = Z.conjugate()*U
    x = 2*XY.real
    y = 2*XY.imag
    z = abs(Z)**2 - abs(U)**2

    return np.array([x, y, z]).T
    
