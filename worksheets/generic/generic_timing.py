import json
import timeit
from hopf.util.matlab_io import save_variables
from numpy import logspace



setup_stmt = """
from hopf.simulation import Simulation
s = Simulation()
s.load_initial_conditions('random40.mat')

"""

times = {}

for integrator in ['rk4', 'sphere-midpoint', 'midpoint', 'lie-poisson']:
    # Statement to time (run simulation but do not save results)
    stmt="s.run_simulation(h=0.1, tmax=10, sim=\'%s\')" % integrator
    t = timeit.Timer(stmt=stmt, setup=setup_stmt)
    times[integrator] = t.timeit(number=1)

json.dump(times, open('data/timings.txt', 'w'))










