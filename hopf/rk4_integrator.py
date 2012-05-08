"Autonomous RK4 integrator adapted to point vortices."

# TODO: Move common functionality between this and the sphere integrator into a common base class, as it is the intention that this class has exactly the same calling conventions as the sphere integrator.

import sys
import numpy as np
from numpy import linalg as LA
from math import floor
from continuous_vortex_system import vortex_rhs


class RK4VortexIntegrator:
    def __init__(self, gamma, sigma=0.0, h=1e-1, verbose=False):
    
        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.h = h
        self.verbose = verbose

    def one_step_rk4(self, X0):
        """Integrate the point vortex equations forward in time by means 
        of a simple RK4 algorithm followed by projection onto the sphere."""

        X1 = X0.copy()

        K1 = vortex_rhs(self.gamma, X1, self.sigma)
        K2 = vortex_rhs(self.gamma, X1 + self.h/2.*K1, self.sigma)
        K3 = vortex_rhs(self.gamma, X1 + self.h/2.*K2, self.sigma)
        K4 = vortex_rhs(self.gamma, X1 + self.h*K3, self.sigma)
        X1 += self.h/6.0*(K1 + 2*K2 + 2*K3 + K4) 
        
        # Project down to the sphere
        for k in xrange(0, X1.shape[0]):
            X1[k, :] /= LA.norm(X1[k, :])
            
        return X1


    def integrate(self, X0, tmax=50., numpoints=100):

        num_inner = int(floor(tmax/(self.h*numpoints)))
        t = 0

        vortices = np.zeros((numpoints, ) + X0.shape)
        times = np.zeros(numpoints)

        if self.verbose:
            print >> sys.stderr, "Entering integration loop"

        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):
                X0 = self.one_step_rk4(X0)
                t += self.h

            # Save output
            vortices[k, :, :] = X0
            times[k] = t

        print >> sys.stderr, '\n'
        return vortices, times
