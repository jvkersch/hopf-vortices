import numpy as np
cimport numpy as np
cimport cython 


DTYPE=np.double
ctypedef np.double_t DTYPE_t


# Specific to certain Lie algebras

# Weyl basis for SO(3)

E0 = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=DTYPE)
E1 = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=DTYPE)
E2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=DTYPE)
# epsilon = np.array([E1, E2, E3], dtype=DTYPE)


cpdef np.ndarray[DTYPE_t, ndim=2] hat(np.ndarray[DTYPE_t, ndim=1] xi):
    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.zeros([3, 3], dtype=DTYPE)
    mat[0, 1] =  xi[2]; mat[1, 0] = -xi[2]
    mat[0, 2] = -xi[1]; mat[2, 0] =  xi[1]
    mat[1, 2] =  xi[0]; mat[2, 1] = -xi[0]
    return mat

cpdef np.ndarray[DTYPE_t, ndim=1] invhat(np.ndarray[DTYPE_t, ndim=2] ximat):
    #cdef np.ndarray[DTYPE_t, ndim=1] xi
    #xi = np.array([ximat[1, 2], ximat[2, 0], ximat[0, 1]], dtype=DTYPE)
    return np.array([ximat[1, 2], ximat[2, 0], ximat[0, 1]], dtype=DTYPE)
