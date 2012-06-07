import sys
import numpy as np
import scipy.optimize as so


# TODO: here there should probably not be any mention of point vortices


class GenericIntegrator:
    """
    Generic numerical integrator functionality.
    """
    def __init__(self, h=1e-1, verbose=False):

        self.h = h
        self.verbose = verbose


    def do_one_step(self, t, X0):
        """Integrate for one timestep with chosen method.  To be overrided."""
        raise NotImplementedError("Method should be overrided in base class.")


    def integrate(self, X0, tmax=50., numpoints=100, **kwargs):

        num_inner = int(round(tmax/(self.h*numpoints)))
        t = 0

        vortices = np.zeros((numpoints, ) + X0.shape)
        times = np.zeros(numpoints)

        if self.verbose:
            print >> sys.stderr, "Entering integration loop"

        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):
                X0 = self.do_one_step(t, X0)
                t += self.h

            # Save output
            vortices[k, :, :] = X0
            times[k] = t


        print >> sys.stderr, '\n'
        return vortices, times
