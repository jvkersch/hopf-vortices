import numpy as np

from ..util.vectors import normalize, is_float_equal, row_product

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
    (1, 2, 2)
    >>> A[0, :, :]
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
    A = np.array(A)
    if A.ndim == 2:
        A = np.array([A])

    N = A.shape[0]
    Z = np.zeros((2, 2))

    out = np.empty(N, dtype=np.bool)
    for k, u in enumerate(A):
        xi = np.matrix(u) 
        out[k] = is_float_equal(xi + xi.H, Z, tol) and abs(np.trace(xi)) < tol

    return out


def is_SU(U, tol=1e-12):
    r"""
    Determine whether the slices along the 0-axis of the array `U` represent 
    elements of the Lie group :math:`SU(2)`, by checking whether
    the relations 

    .. math::

        UU^\dagger = U^\dagger U = \mathrm{I}, \quad \det(U) = 1

    hold for each slice, up to a tolerance specified by `tol`.
    
    Parameters
    ----------

    U : array_like
        Nx2x2 complex array.
    tol : float, optional
          Tolerance with which to verify :math:`SU(2)`-criteria.
   
    Returns
    -------

    out : array-like
          N-vector of booleans determining whether the corresponding slice of `U`
          belongs to :math:`SU(2)`.

    Examples
    --------

    >>> A = hatmap([1, 2, 3]); A
    array([[[ 0.+3.j,  2.+1.j],
            [-2.+1.j, -0.-3.j]]])
    >>> import scipy.linalg
    >>> U = scipy.linalg.expm(A[0, :, :]); U
    array([[-0.82529906-0.45276398j, -0.30184265-0.15092133j],
           [ 0.30184265-0.15092133j, -0.82529906+0.45276398j]])
    >>> is_SU(U)
    array([ True], dtype=bool)

    """

    U = np.array(U)
    if U.ndim == 2:
        U = np.array([U])

    N = U.shape[0]
    I = np.eye(2)

    out = np.empty(N, dtype=np.bool)
    for k, u in enumerate(U):
        um = np.matrix(u) 
        out[k] = is_float_equal(um*um.H, I, tol) and \
            abs(np.linalg.det(um)-1) < tol

    return out


def cayley_klein(a):
    r"""
    Apply the explicit form of the :math:`\mathfrak{su}(2)`-Cayley map to 
    the rows of A.

    Parameters
    ----------

    a : array-like
        Nx3 real array of :math:`\mathfrak{su}(2)`-elements in the vector 
        representation.

    Returns
    -------

    out: array-like
         Nx2x2 complex array of :math:`SU(2)`-elements.

    Examples
    --------

    >>> a = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> U = cayley_klein(a); U
    array([[[-0.86666667+0.4j       ,  0.26666667+0.13333333j],
            [-0.26666667+0.13333333j, -0.86666667-0.4j       ]],
    <BLANKLINE>
           [[-0.97435897+0.15384615j,  0.12820513+0.1025641j ],
            [-0.12820513+0.1025641j , -0.97435897-0.15384615j]]])
    >>> is_SU(U)
    array([ True,  True], dtype=bool)

    """

    a = np.array(a)
    if a.ndim == 1:
        a = np.array([a])

    N = a.shape[0]
    A = hatmap(a)
    norms2 = np.sum(a*a, axis=1)

    U = np.zeros((N, 2, 2))
    U[:, 0, 0] = np.ones(N)
    U[:, 1, 1] = np.ones(N)

    U = np.einsum('a, abc -> abc', 1-norms2, U) + 2*A
    U = np.einsum('a, abc -> abc', 1./(1+norms2), U)

    return U


def apply_2by2(U, psi):
    r"""Apply the matrices in `U' to the vectors in `psi' along axis 0.

    Examples
    --------

    Check output with direct loop computation:

    >>> U = np.empty((3, 2, 2), np.double)
    >>> U[0, :, :] = [[1, 2], [3, 4]]
    >>> U[1, :, :] = [[10, 11], [12, 13]]
    >>> U[2, :, :] = [[20, 21], [22, 23]]
    >>> psi = np.array([[30, 29], [28, 27], [26, 25]], np.double)
    array([[   88.,   206.],
           [  577.,   687.],
           [ 1045.,  1147.]])
    >>> psi2 = np.empty((3, 2), np.double)
    >>> for k in xrange(0, 3): psi2[k, :] = np.dot(U[k, :, :], psi[k, :])
    >>> psi2 
    array([[   88.,   206.],
           [  577.,   687.],
           [ 1045.,  1147.]])

    """

    return np.einsum('abc, ac -> ab', U, psi)



def cayley_klein_vectorial(a, x):
    r"""Apply the Cayley-Klein map directly to a set of vectors `x`.

    TODO: might be slow

    TODO: test this

    Examples
    --------

    >>> a = [[1, 2, 3], [4, 5, 6]]
    >>> x = [[7, 8, 9], [10, 11, 12]]
    >>> cayley_klein(a, x)
    array([[  3.90666667,  10.34666667,   8.46666667],
           [  9.56607495,  11.58382643,  11.80276134]])

    """
    a = np.array(a)
    x = np.array(x)

    if a.ndim == 1:
        a = np.array([a])
        x = np.array([x])

    norms2 = np.sum(a*a, axis=1)
    dot_ax = np.sum(a*x, axis=1)

    tmp = row_product((1.-norms2)**2, x) - \
        4.*row_product(1. - norms2, np.cross(a, x, axis=1)) + \
        8.*row_product(dot_ax, a) - \
        4.*row_product(norms2, x)

    tmp = row_product(1./(1. + norms2)**2, tmp)
    return normalize(tmp)


def hopf(psi):
    """Apply the Hopf fibration map to the rows of `psi`.

    Parameters
    ----------

    psi : array-like
          Nx2 complex array whose rows are unit vectors on the three-sphere.

    Returns
    -------

    X : array-like
        Nx3 real array of Hopf-projected vectors.

    Examples
    --------

    >>> hopf([1j, 0])
    array([[ 0.,  0.,  1.]])
    >>> hopf([1, 0])
    array([[ 0.,  0.,  1.]])

    """
    psi = np.array(psi, dtype=complex)
    if psi.ndim == 1:
        psi = np.array([psi])

    Z = psi[:, 0]
    U = psi[:, 1]

    XY = Z.conjugate()*U
    x = 2*XY.real
    y = 2*XY.imag
    z = abs(Z)**2 - abs(U)**2

    return np.array([x, y, z]).T
    

def inverse_hopf(X):
    """
    Apply inverse of the Hopf map, i.e. finds an element in the 
    fiber of the Hopf map over a given point.

    Parameters
    ----------

    X : array-like
        Nx3 real array of vectors on the 2-sphere.

    Returns 
    -------

    psi : array-like
          Nx2 complex array of vectors on the 3-sphere that project down onto 
          the rows of `X`.

    Examples
    --------

    >>> X = hopf([1., 0])
    >>> inverse_hopf(X)
    array([[ 1.+0.j,  0.+0.j]])

    Note: due to non-uniqueness of the inverse Hopf map, the composition 
    of `inverse_hopf` with `hopf` will only give the identity up to a 
    complex phase.

    >>> X = hopf([1j, 0])
    >>> inverse_hopf(X)
    array([[ 1.+0.j,  0.+0.j]])

    """

    X = np.array(X)
    if X.ndim == 1:
        X = np.array([X])

    n, m = X.shape
    x = X[:, 0]; y = X[:, 1]; z = X[:, 2]

    f = 1./(1. + abs(z))
    zeta = f*(x + 1j*y)
    zeta_conj = np.vstack((zeta.conjugate(), zeta)).T

    psi = np.ones((n, 2), dtype=complex)
    
    logical_index = np.vstack((z<=0, z>0)).T
    psi[logical_index] = zeta_conj[logical_index]
    psi = normalize(psi)

    return psi


