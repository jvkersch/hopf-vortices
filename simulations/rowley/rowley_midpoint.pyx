from __future__ import division

import sys

import numpy as np
cimport numpy as np
cimport cython 

from math import pi, log
cdef double PI = pi

from numpy.linalg import norm

from scipy.optimize import fsolve
from realify import Realify, C_to_R, R_to_C


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef double hamiltonian(
    np.ndarray[np.double_t, ndim=1] gamma, 
    np.ndarray[double complex, ndim=1] z):
    """
    Computes the point vortex Hamiltonian.

    """

    cdef double H = 0
    cdef int i, j
    cdef int N = z.shape[0]

    for i from 0 <= i < N:
        for j from i < j < N:
            H += gamma[i]*gamma[j]/(4*PI)*log(norm(z[i]-z[j]))

    return H


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef np.ndarray[double complex, ndim=1] f(
    np.ndarray[np.double_t, ndim=1] gamma, 
    np.ndarray[double complex, ndim=1] z):
    """
    Computes the complex conjugate of the RHS of the point vortex equations.
    
    INPUT
    -----
    
     - `gamma` -- real vector of vortex strengths
     - `z` -- complex vector of vortex locations

    OUTPUT
    ------

    Complex vector of RHS entries.

    ..note::

    Timing information:

    >>> %timeit -n100 -r20 f(gamma, z)
    100 loops, best of 20: 39.8 us per loop

    """

    cdef int j, l
    cdef int N = z.shape[0]

    cdef double complex diff
    cdef np.ndarray res = np.zeros(N, dtype=np.complex)

    for j from 0 <= j < N:
        for l from j < l < N:
            diff = 1/(2.*PI*1j*(z[j]-z[l]))

            res[j] += gamma[l]*diff
            res[l] -= gamma[j]*diff

    return res.conjugate()


def rowley_one_step(gamma, z0, z1, h, sigma):
    """
    Takes one step with the Rowley midpoint algorithm.

    INPUT 
    -----

    """
    
    z_a = (1-sigma)*z0 + sigma*z1
    f_a = f(gamma, z_a)

    def nonlinear_eq(z2):
        z_b = (1-sigma)*z1 + sigma*z2
        return z2 - z0 - 2*h*(sigma*f_a + (1-sigma)*f(gamma, z_b))

    n_eq_real = Realify(nonlinear_eq)
    return R_to_C(fsolve(n_eq_real, C_to_R(z1)))
    

def vortex_rk4(gamma, z0, h):
    """
    Numerically solve the point vortex equations using RK4.

    """

    rhs = lambda gamma, z: f(gamma, z)

    k1 = rhs(gamma, z0)
    k2 = rhs(gamma, z0 + h/2.*k1)
    k3 = rhs(gamma, z0 + h/2.*k2)
    k4 = rhs(gamma, z0 + h*k3)

    z1 = z0 + (k1 + 2*k2 + 2*k3 + k4)*h/6.
    return z1


def rowley_integrate(gamma, z0, tmax=50., numpoints=100, h=.1, sigma=.5):
    """
    Integrate point vortex equations using Rowley's algorithm
    over a definite length of time.

    """

    num_inner = int(round(tmax/(h*numpoints)))

    zs = np.zeros((numpoints, ) + z0.shape, dtype=np.complex)
    times = np.zeros(numpoints)

    z1 = vortex_rk4(gamma, z0, h)
    t  = h

    for k in xrange(0, numpoints):
        print >> sys.stderr, '.',

        for _ in xrange(0, num_inner):
            z2 = rowley_one_step(gamma, z0, z1, h, sigma)
            #z2 = vortex_rk4(gamma, z1, h) 
            z1, z0 = z2, z1
            t += h

        zs[k, :, ] = z1
        times[k] = t

    print >> sys.stderr, '\n'
    return zs, times


def leapfrog_initial_conditions():
    """
    Set up initial conditions for 4 leapfrogging vortices.

    """

    gamma = np.array([1., 1., -1., -1.])
    z0 = np.array([-1.+2.j, 1.+2.j, -1.-2.j, 1.-2.j], dtype=np.complex)
    return gamma, z0

def random_initial_conditions(xmax=1., ymax=1., n=4):
    """
    Set up random initial conditions for `n` vortices.

    """
    from numpy.random import random_sample
    gamma = random_sample(n)
    x = xmax*random_sample(n)
    y = ymax*random_sample(n)

    return gamma, x + 1j*y
