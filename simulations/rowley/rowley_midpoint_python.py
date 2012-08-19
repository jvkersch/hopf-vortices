from __future__ import division

import numpy as np
from math import pi


def f(gamma, z):
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
    100 loops, best of 20: 101 us per loop


    """

    N = z.shape[0]
    res = np.zeros(N, dtype=np.complex)

    for j in xrange(0, N):
        for l in xrange(j+1, N):
            diff = gamma[l]/(2.*pi*1j*(z[j]-z[l]))

            res[j] += diff
            res[l] -= diff

    return res.conjugate()
