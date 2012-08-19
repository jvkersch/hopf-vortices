"""
Quick-n-dirty way of turning a function from C^n to C^n into a 
function from RR^2n to RR^2n.

"""

import numpy as np

def C_to_R(z):
    """
    Turn complex n-vector into real vector of length 2n
    
    """
    n = z.size
    v = np.empty((2*n,))
    v[0::2] = z.real
    v[1::2] = z.imag
    return v


def R_to_C(v):
    """
    Turn real vector of length 2n into complex n-vector.
    
    """
    x = v[0::2]
    y = v[1::2]
    return x + 1j*y


class Realify:
    def __init__(self, complex_fun):
        self.complex_fun = complex_fun


    def __call__(self, v):
        """
        Wrapper for complex-valued function.

        """
        return C_to_R(self.complex_fun(R_to_C(v)))


if __name__ == '__main__':

    f = lambda z: z**2
    r = Realify(f)

    print r(np.array([3., 4.]))
