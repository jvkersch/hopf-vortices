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
    """
    Applies hatmap from R3 to su(2) row-wise.

    Parameters
    ----------

    a : array-like
        Nx3 array of real entries, or 3-vector.

    Returns 
    -------

    A : array-like
        Complex Nx2x2 array whose slices along the 2-axis
        are the corresponding elements of su(2).

    Examples
    --------

    >>> A = hatmap([1, 2, 3])
    >>> A.shape
    (2, 2, 1)
    >>> A[:, :, 0]
    array([[ 0.+3.j,  2.+1.j],
    [-2.+1.j, -0.-3.j]])

    """
    a = np.array(a)
    if a.ndim == 1:
        a = np.array([a])

    N = a.shape[0]

    x = np.array(a[:, 0])
    y = np.array(a[:, 1])
    z = np.array(a[:, 2])

    A = np.empty((N, 2, 2), dtype=complex)
    A[:, 0, 0] = z*1j
    A[:, 0, 1] =  y + x*1j
    A[:, 1, 0] = -y + x*1j
    A[:, 1, 1] = -z*1j

    return A


def inverse_hatmap(A):
    """
    Apply the inverse of the hatmap to the su(2)-slices along the 0-axis of A. In other words, 
    return the vector representation of the slices of A.

    Parameters
    ----------

    A : array-like
        Complex Nx2x2 array whose slices along the 0-axis are elements of su(2), 
        or complex 2x2 array representing an element of su(2).

    Returns
    -------

    a : array-like
        Real Nx3 array containing the vector representation of the slices of A.

    Examples
    --------

    >>> A = np.array([[3j, 2+1j], [-2+1j, -3j]]); A
    array([[ 0.+3.j,  2.+1.j],
       [-2.+1.j,  0.-3.j]])
    >>> a = inverse_hatmap(A); a
    array([[ 1.,  2.,  3.]])

    """
    
    A = np.array(A)
    if A.ndim == 2:
        A = np.array([A])

    x =  A[:, 0, 1].imag
    y =  A[:, 0, 1].real
    z = (A[:, 0, 0]/(1j)).real

    return np.array([x, y, z]).T


def is_su(A, tol=1e-12):
    r"""
    Determine whether the slices along the 0-axis of the array `A` represent 
    elements of the Lie algebra :math:`\mathfrak{su}(2)`, by checking whether
    the relations 

    .. math::

        A = -A^\dagger, \quad \mathrm{trace}(A) = 0

    hold for each slice, up to a tolerance specified by `tol`.

    Parameters
    ----------

    A : array-like
        Nx2x2 array of complex numbers.
    tol : float, optional
        Tolerance with which to verify :math:`\mathfrak{su}(2)`-criteria.
   
    Returns
    -------

    out : array-like
          N-vector of booleans determining whether the corresponding slice of `A`
          belongs to :math:`\mathfrak{su}(2)`.

    Examples
    --------

    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> A = hatmap(a)
    >>> is_su(A)
    array([ True,  True], dtype=bool)

    """
    if A.ndim == 2:
        A = np.array([A])

    N = A.shape[0]
    Z = np.zeros((2, 2))

    out = np.empty(N, dtype=np.bool)
    for k, u in enumerate(A):
        xi = np.matrix(u) 
        out[k] = is_float_equal(xi + xi.H, tol) and abs(np.trace(xi)) < tol

    return out

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




def cayley_klein(a):
    """Applies explicit form of the Cayley map in su(2) to the rows of A."""

    if a.ndim == 1:
        a = np.array([a])
    else:
        a = np.array(a)

    N = a.shape[0]
    I = np.eye(2)

    A = hatmap(a)
    norms2 = np.sum(a*a, axis=1)

    U = np.ones((2, 2, N))
    U = np.tensordot(1-norms2, U, axes=(0, 0)) + 2*A
    U = np.tensordot(1/(1-norms2), U, axes=(0, 0))

    return U


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


