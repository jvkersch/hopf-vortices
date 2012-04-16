import numpy as np

from vectors import normalize, is_float_equal

# Pauli matrices
pauli = np.empty((2, 2, 3), dtype=complex)
pauli[:, :, 0] = [[0, 1], [1, 0]]
pauli[:, :, 1] = [[0, -1j], [1j, 0]]
pauli[:, :, 2] = [[1, 0], [0, -1]]

# Standard basis for Lie algebra su(2)
tau_basis = 1j*pauli


def hatmap(a):
    """Applies hatmap from R3 to su(2) row-wise.

    INPUT: 

      -- ``a`` - Nx3 real array.

    OUTPUT:

    Complex 2x2xN array whose slices along the 2-axis
    are the corresponding elements of su(2).

    """

    x = np.array(a[:, 0])
    y = np.array(a[:, 1])
    z = np.array(a[:, 2])

    A = np.array([[z*1j, y + x*1j], [-y + x*1j, -z*1j]])
    return A


def inverse_hatmap(A):

    x = A[0, 1, :].imag
    y = A[0, 1, :].real
    z = (A[0, 0, :]/(1j)).real

    return np.array([x, y, z]).T


def is_su(A, tol=1e-12):

    if A.ndim == 2:
        A = np.array([A])

    n = A.shape[2]
    for k in xrange(0, n):
        xi = np.matrix(A[:, :, k])
        if not is_float_equal(xi + xi.H, np.zeros((2, 2)), tol) or \
           not abs(np.trace(xi)) < tol:
            return False

    return True



def is_SU(U, tol=1e-12):
    """Tests whether a given matrix belongs to SU(N), i.e. satisfies 
    :math:`UU^\dagger = U^\dagger U = I` and :math:`\det(U) = 1`.

    TODO: docstring out of date.

    INPUT:
    
      -- ``U`` - square matrix.

      -- ``tol`` - tolerance with which to verify unitarity.

    """

    if U.ndim == 2:
        U = np.array([U])

    n = U.shape[0]
    U = np.array(U)
    I = np.eye(n)

    for k in xrange(0, n):
        U_mat = np.matrix(U[:, :, k])
        if not is_float_equal(np.dot(U_mat, U_mat.H), I) or \
           not abs(np.linalg.det(U_mat)-1) < tol:
            return False

    return True


def hopf(psi):
    """Applies Hopf fibration row-wise.

    INPUT:

      -- ``psi`` - complex Nx2 array with unit vectors on three-sphere.

    OUTPUT:

      Real-valued Nx3 array of corresponding 3-vectors on two-sphere.

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


