import numpy as np
cimport numpy as np
cimport cython 

from ..lie_algebras.so3_geometry import hat, invhat

DTYPE=np.double
ctypedef np.double_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] vector_invhat(
    np.ndarray[DTYPE_t, ndim=3] rho, np.ndarray[DTYPE_t, ndim=1] gamma):

    cdef int N = rho.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] x = np.empty((N, 3))
    cdef int n

    for n from 0 <= n < N:
        x[n, :] = invhat(rho[n, :, :])/gamma[n]

    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=3] vector_hatmap(
    np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] gamma):

    cdef int N = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] rho = np.empty((N, 3, 3))
    cdef int n

    for n from 0 <= n < N:
        rho[n, :, :] = hat(x[n, :])*gamma[n]

    return rho
