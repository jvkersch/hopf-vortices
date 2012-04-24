setup = """
from vortex_integrator import VortexIntegrator
from matlab_io import load_ic_from_matfile 
gamma, x0, sigma = load_ic_from_matfile('collapse3py.mat')
v = VortexIntegrator(gamma, sigma=0.0, h=1e-1, verbose=False)
"""
   
stmt  = "v.integrate(x0)"

import timeit
t = timeit.Timer(stmt=stmt, setup=setup)
print t.timeit(number=3)
