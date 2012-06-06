"Autonomous RK4 integrator adapted to point vortices."

import sys
import numpy as np
from numpy import linalg as LA
from math import floor

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs


class RK4VortexIntegrator(GenericIntegrator):
    def __init__(self, gamma, sigma=0.0, h=1e-1, verbose=False):
    
        self.gamma = np.array(gamma)
        self.sigma = sigma

        GenericIntegrator.__init__(self, h, verbose)

    def do_one_step(self, t, X0):
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

