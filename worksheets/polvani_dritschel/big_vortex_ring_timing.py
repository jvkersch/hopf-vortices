import json
import timeit
from hopf.util.matlab_io import save_variables
from numpy import logspace


N = 10

setup_stmt = """
from hopf.simulation import Simulation
s = Simulation()
s.load_initial_conditions('equatorial_ring_40.mat')
s.sigma = 0.10

"""
integrators = ['rk4', 'sphere-midpoint', 'midpoint', 'lie-poisson']
times = {integrator: [] for integrator in integrators}

for _ in xrange(0, N):
    # Do N runs with timings
    for integrator in integrators:
        # Statement to time (run simulation but do not save results)
        stmt="s.run_simulation(h=0.1, tmax=50, sim=\'%s\')" % integrator
        t = timeit.Timer(stmt=stmt, setup=setup_stmt)
        times[integrator].append( t.timeit(number=1) )

json.dump(times, open('data/timings.txt', 'w'))










