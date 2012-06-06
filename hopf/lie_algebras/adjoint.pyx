import numpy as np
cimport numpy as np
cimport cython 


DTYPE=np.double
ctypedef np.double_t DTYPE_t

# General-purpose code to deal with matrix groups.

cpdef np.ndarray[DTYPE_t, ndim=2] Ad(
    np.ndarray[DTYPE_t, ndim=2] g,
    np.ndarray[DTYPE_t, ndim=2] xi):
    return np.dot(g, np.dot(xi, np.linalg.inv(g)))


cpdef np.ndarray[DTYPE_t, ndim=3] AdN(
    np.ndarray[DTYPE_t, ndim=3] g,
    np.ndarray[DTYPE_t, ndim=3] xi):
    """Vectorized Ad-mapping."""

    cdef int n
    cdef np.ndarray[DTYPE_t, ndim=2] out 

    out = np.empty((xi.shape[0], xi.shape[1], xi.shape[2]))

    for n from 0 <= n < xi.shape[0]:
        out[n, :, :] = Ad(g[n, :, :], xi[n, :, :])

    return out


