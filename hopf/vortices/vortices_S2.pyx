"""
Functions dealing with continuous-time vortex systems.

"""


import numpy as np
cimport numpy as np
cimport cython 


DTYPE=np.double
ctypedef np.double_t DTYPE_t

from math import pi
cdef double PI = pi

cdef extern from "math.h":
    double log(double)


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] gradient_S2_slow(
    np.ndarray[DTYPE_t, ndim=1] gamma, 
    np.ndarray[DTYPE_t, ndim=2] X, 
    DTYPE_t sigma):
    """
    Scaled gradient of the point vortex Hamiltonian, where the Hamiltonian
    has been written in terms of inner products between the vortex 
    locations.

    """

    cdef int N = X.shape[0]
    cdef int ndim = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros([N, ndim], dtype=DTYPE)
    cdef int i, j
    cdef double d

    for i from 0 <= i < N:
        for j from 0 <= j < N:
            if i == j: continue
            d = sigma**2 + 1 - np.dot(X[i, :], X[j, :])
            res[i, :] += 1/(4.*PI)*gamma[j]*X[j, :]/d

    return res

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] gradient_S2(
    np.ndarray[DTYPE_t, ndim=2] gradient,
    np.ndarray[DTYPE_t, ndim=1] gamma, 
    np.ndarray[DTYPE_t, ndim=2] X, 
    DTYPE_t sigma):
    """
    Scaled gradient of the point vortex Hamiltonian, where the Hamiltonian
    has been written in terms of inner products between the vortex 
    locations.

    """

    cdef int N = X.shape[0]
    cdef int ndim = X.shape[1]
    cdef int i, j, alpha
    cdef double d

    for i from 0 <= i < N:
        for alpha from 0 <= alpha < 3:
            gradient[i, alpha] = 0
        for j from 0 <= j < N:
            if i == j: continue
            d = sigma**2 + 1
            for alpha from 0 <= alpha < 3:
                d -= X[i,alpha] * X[j,alpha]
            for alpha from 0 <= alpha < 3:
                gradient[i, alpha] += 1/(4.*PI)*gamma[j]*X[j, alpha]/d


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] gradient_S2_old(
    np.ndarray[DTYPE_t, ndim=1] gamma, 
    np.ndarray[DTYPE_t, ndim=2] X, 
    DTYPE_t sigma):
    """
    Scaled gradient of the Hamiltonian defined in terms of lengths of 
    differences between vectors, rather than inner products. Agrees 
    with previous implementations of the gradient on the unit sphere.
    
    """
    cdef int N = X.shape[0]
    cdef int ndim = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros([N, ndim], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] vec 
    cdef int i, j

    for i from 0 <= i < N:
        for j from i+1 <= j < N:
            vec = X[i, :] - X[j, :]
            vec /= np.dot(vec, vec) + 2*sigma**2
            res[i, :] += gamma[j]*vec
            res[j, :] -= gamma[i]*vec
        res[i, :] *= -1/(2.*PI)

    return res


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] vortex_rhs(
    np.ndarray[DTYPE_t, ndim=1] gamma, 
    np.ndarray[DTYPE_t, ndim=2] X,
    DTYPE_t sigma):
    """
    Computes the RHS of the point vortex equations.

    """

    cdef np.ndarray[DTYPE_t, ndim=2] res
    cdef int i

    res = gradient_S2_slow(gamma, X, sigma)

    for i from 0 <= i < X.shape[0]:
        res[i, :] = np.cross(res[i, :], X[i, :])

    return res


@cython.boundscheck(False) 
@cython.wraparound(False)
def hamiltonian_S2(np.ndarray[DTYPE_t, ndim=1] gamma, 
                   np.ndarray[DTYPE_t, ndim=2] X,
                   DTYPE_t sigma):
    """
    Value of the energy for a vortex configuration, computed using 
    inner products.

    INPUT:

      - ``gamma`` - Vector of vortex strengths.

      - ``X`` - Array of vortex locations in 3D.

      - ``sigma`` - Scalar value for regularization parameter.

    OUTPUT:

    Value of the energy as a scalar.

    """

    cdef int N = X.shape[0]
    cdef int ndim = X.shape[1]
    cdef int i, j
    cdef DTYPE_t E = 0

    for i from 0 <= i < N:
        for j from i+1 <= j < N:
            factor = 2*sigma**2 + 2*(1 - np.dot(X[i, :], X[j, :]))
            E -= gamma[i]*gamma[j]/(4*PI)*log(factor)
             
    return E


@cython.boundscheck(False) 
@cython.wraparound(False)
def hamiltonian_S2_old(np.ndarray[DTYPE_t, ndim=1] gamma, 
                       np.ndarray[DTYPE_t, ndim=2] X,
                       DTYPE_t sigma):
    """Value of the energy for a vortex configuration.

    INPUT:

      - ``gamma`` - Vector of vortex strengths.

      - ``X`` - Array of vortex locations in 3D.

      - ``sigma`` - Scalar value for regularization parameter.

    OUTPUT:

    Value of the energy as a scalar.

    """

    cdef int N = X.shape[0]
    cdef int ndim = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] vec 
    cdef int i, j
    cdef DTYPE_t E = 0

    for i from 0 <= i < N:
        for j from i+1 <= j < N:
            vec = X[i, :] - X[j, :]
            E -= gamma[i]*gamma[j]/(4*PI)*log(2*sigma**2 + np.dot(vec, vec))
             
    return E


@cython.boundscheck(False) 
@cython.wraparound(False)
def momentum_S2(np.ndarray[DTYPE_t, ndim=1] gamma, 
                np.ndarray[DTYPE_t, ndim=2] X):
    """Moment of a vortex configuration.

    INPUT:

      - ``gamma`` - Vector of vortex strengths.

      - ``X`` - Array of vortex locations in 3D.

    OUTPUT:

    Value of the vortex moment as a numpy three-vector.
      
    """

    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] m = np.empty((3, ), dtype=DTYPE)

    for i from 0 <= i < 3:
        m[i] = np.sum(gamma*X[:, i])

    return m




