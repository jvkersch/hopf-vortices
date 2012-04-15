"""
Utility functions to deal with vectors in n-dimensional space.

"""
import numpy as np


# Utility functions

def hstack(v, n):
    """Return array with vector v in each of the n columns"""
    return np.kron(np.ones((n, 1)), v).T


# Linear algebra

def is_float_equal(v1, v2, tol=1e-7):
    "Equality of floating point vectors."
    return (np.abs(v1 - v2) < tol).all()

def normalize(array):
    """Normalize rows of array to unit length. Array can be real or complex."""

    norms = np.sqrt(np.sum(array*array.conjugate(), axis=1))
    return array/hstack(norms, array.shape[1])


# Random vectors and matrices

def random_vector(length, iscomplex=False):
    """Return a random vector of a given length.

    INPUT:

      -- ``length`` - length of the random vector.
      -- ``iscomplex`` (default `False`) - whether to return a complex vector.

    """
    size = (1, length) 
    return random_array(size, iscomplex)[0]


def random_array(size, iscomplex=False):
    """Return a random matrix of a given length.

    INPUT: 
    
      -- ``size`` - tuple containing size of the array.
      -- ``iscomplex`` (default `False`) - whether to return a complex array.

    """
    
    A = np.random.rand(*size)
    if iscomplex: A = A + 1j*np.random.rand(*size)
    return A
