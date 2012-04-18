import numpy as np

def cayley(A):
    r"""
    Apply Cayley map to elements of a matrix Lie algebra. 

    Parameters
    ----------

    A : array-like
        Nxmxm array whose slices along the 0-axis are Lie algebra elements.

    Returns
    -------

    M : array-like
        Nxmxm array whose slices are Lie group elements.

    Examples
    --------

    >>> from su2_geometry import hatmap
    >>> A = hatmap([1, 2, 3]); A
    array([[[ 0.+3.j,  2.+1.j],
            [-2.+1.j, -0.-3.j]]])
    >>> cayley(A)
    array([[[-0.86666667+0.4j       ,  0.26666667+0.13333333j],
            [-0.26666667+0.13333333j, -0.86666667-0.4j       ]]])

    This Cayley map agrees with the `cayley_klein` for matrices in 
    :math:`\mathfrak{su}(2)`:

    >>> from su2_geometry import cayley_klein
    >>> cayley_klein([1, 2, 3])
    array([[[-0.86666667+0.4j       ,  0.26666667+0.13333333j],
            [-0.26666667+0.13333333j, -0.86666667-0.4j       ]]])

    """

    A = np.array(A)
    if A.ndim == 2:
        A = np.array([A])

    N, m, _ = A.shape
    I = np.eye(m)
    M = np.empty((N, m, m), dtype=A.dtype)

    for k in xrange(0, N):
        element = A[k, :, :]
        M[k, :, :] = np.dot(I + element, np.linalg.inv(I - element))

    return M
