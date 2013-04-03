import numpy as np
cimport numpy as np
cimport cython 

# TODO clean up import mess below

# Why doesn't this work?
#from libc.math import pi, log

#DTYPE=np.complex
ctypedef np.complex complex_t

cdef double PI = 3.14159265359

#cdef extern from "math.h":
#    double log(double)

# Is this in libc?
cdef extern from "complex.h":
    complex_t conj(complex_t z)
    double    real(complex_t z)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef complex_t pairwise_hermitian_product(complex_t[:,::1] psi, 
                                          int row1, int row2):

    """
    Hermitian inner product of complex vectors using direct subscripting.

    """
    cdef complex_t p
    cdef int i, N

    p = 0.
    N = psi.shape[1]

    for i from 0 <= i < N:
        p += conj(psi[row1, i]) * psi[row2, i]

    return p


@cython.boundscheck(False) 
@cython.wraparound(False)
def scaled_gradient_hamiltonian_S3(complex_t[:, ::1] gradient,
                                   double[::1] gamma, 
                                   complex_t[:, ::1] psi, 
                                   double sigma):

    """
    Optimized version of the scaled version of the gradient of the 
    S3-Hamiltonian, evaluated at point vortex locations `psi`. 

    The output is stored in `gradient`.

    """

    cdef int k, l, m
    cdef int N
    cdef complex_t p_ll, p_kk, p_kl, p_lk
    cdef complex_t denominator
    
    N = psi.shape[0]

    for k from 0 <= k < N:
        gradient[k, 0] = 0
        gradient[k, 1] = 0
        
        for l from 0 <= l < N:
            if l == k: continue
            p_ll = pairwise_hermitian_product(psi, l, l)
            p_kk = pairwise_hermitian_product(psi, k, k)
            p_lk = pairwise_hermitian_product(psi, l, k)
            p_kl = conj(p_lk)

            denominator = 2*sigma**2 + 2*(1 + p_ll*p_kk - 2*p_lk*p_kl)

            for m from 0 <= m < 2:
                gradient[k, m] -= (gamma[l]/(4*PI*denominator) *
                                   (2*p_ll*psi[k, m] - 4*p_lk*psi[l, m]) )

