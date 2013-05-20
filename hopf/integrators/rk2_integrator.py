"Heun's method, adapted to the point vortex system."

import sys
import numpy as np
from numpy import linalg as LA
from math import floor

from generic_integrator import GenericIntegrator
from ..vortices.vortices_S2 import vortex_rhs


class RK2VortexIntegrator(GenericIntegrator):
    def __init__(self, gamma, sigma=0.0, h=1e-1, verbose=False):
    
        self.gamma = np.array(gamma)
        self.sigma = sigma

        GenericIntegrator.__init__(self, h, verbose)

    def do_one_step(self, t, X0):
        """Integrate the point vortex equations forward in time by means 
        of a simple RK2 algorithm followed by projection onto the sphere."""

        #X1 = X0.copy()
        
        rhs = vortex_rhs(self.gamma, X0, self.sigma)

        Y  = X0 + self.h * rhs
        X1 = X0 + self.h/2 * ( rhs + vortex_rhs(self.gamma, Y, self.sigma) )
        
        # Project down to the sphere
        for k in xrange(0, X1.shape[0]):
            X1[k, :] /= LA.norm(X1[k, :])
            
        return X1

