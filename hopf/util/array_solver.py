import scipy.optimize as so
import numpy as np

np.set_printoptions(precision=20)

class FSolveArray:
    """
    Solve nonlinear equation where variables are 2D numpy arrays.

    Examples
    --------

    >>> def fun(x0, A):
    ...     return A*x0 - np.cos(x0)
    ...
    >>> x0 = np.array([[1., 0.5], [.3, .6]])
    >>> A  = np.array([[1, 3], [2, 1]])
    >>> f = FSolveArray(fun, size=(2,2))
    >>> f.fsolve(x0, args=A, xtol=1e-5)
    array([[ 0.7390852 ,  0.31675081],
           [ 0.4501836 ,  0.73908516]])

    """

    def __init__(self, fun, size):

        n, m = size
        self.size = (n, m)
        self.output_size = n*m

        def fun_flattened(x0, *params):
            x0.shape = self.size 
            res = fun(x0, *params)
            res.shape = self.output_size
            return res

        self.fun_flattened = fun_flattened

    def fsolve(self, x0, **kwargs):
        x0 = np.array(x0)
        x0.shape = self.output_size

        res, infodict, ier, msg = so.fsolve(self.fun_flattened, x0, 
                                            full_output=True, **kwargs)
        #print np.max(np.abs(infodict['fvec']))

        res.shape = self.size
        return res

