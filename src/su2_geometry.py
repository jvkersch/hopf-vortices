import numpy as np

from vectors import normalize, is_float_equal


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
    

def inverse_hopf(X):
    """Applies inverse of the Hopf map, i.e. finds an element in the 
    fiber of the Hopf map over a given point.

    INPUT:

      -- ``X`` - Nx3 array of vectors on the 2-sphere.

    OUTPUT: 

    Complex Nx2 array of vectors on the 3-sphere that project down onto X.

    """

    n, m = X.shape
    x = X[:, 0]; y = X[:, 1]; z = X[:, 2]

    f = 1/(1 + abs(z))
    zeta = f*(x + 1j*y)
    zeta_conj = np.vstack((zeta.conjugate(), zeta)).T

    psi = np.ones((n, 2), dtype=complex)
    
    logical_index = np.vstack((z<=0, z>0)).T
    psi[logical_index] = zeta_conj[logical_index]
    psi = normalize(psi)

    return psi


def is_special_unitary(U, tol=1e-12):
    """Tests whether a given matrix belongs to SU(N), i.e. satisfies 
    :math:`UU^\dagger = U^\dagger U = I` and :math:`\det(U) = 1`.

    INPUT:
    
      -- ``U`` - square matrix.
      -- ``tol`` - tolerance with which to verify unitarity.

    """
    I = np.eye(U.shape[0])
    return is_float_equal(np.dot(U, U.H), I) && (abs(np.linalg.det(U)-1) < tol)
