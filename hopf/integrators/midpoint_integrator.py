"Midpoint + projection integrator adapted to point vortices."

import sys
import numpy as np
from numpy import linalg as LA
from math import floor

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs

from scipy.optimize import broyden1


class MidpointIntegrator(GenericIntegrator):
    def __init__(self, gamma, sigma=0.0, h=1e-1, verbose=False):
    
        self.gamma = np.array(gamma)
        self.sigma = sigma

        GenericIntegrator.__init__(self, h, verbose)

    def do_one_step(self, t, X0):

        def optimization_function(X1):
            return X1 - X0 - self.h*vortex_rhs(self.gamma,(X0+X1)/2,self.sigma)

        X1 = broyden1(optimization_function, X0)

        # Project down to the sphere
        for k in xrange(0, X1.shape[0]):
            X1[k, :] /= LA.norm(X1[k, :])
            
        return X1

