import json
import timeit

setup_stmt = """
from hopf.simulation import Simulation
s = Simulation()
s.load_initial_conditions('random40.mat')

"""

times = {}

for h, integrator in [(.03, 'rk4'), (.05, 'sphere-midpoint'), 
                      (0.3, 'midpoint'), (0.1, 'lie-poisson')]:
    # Statement to time (run simulation but do not save results)
    stmt="s.run_simulation(h=%f, tmax=10, sim=\'%s\'); s.post_process(); s.save_results('data/generic_%s.mat')" % (h, integrator, integrator)
    t = timeit.Timer(stmt=stmt, setup=setup_stmt)
    times[integrator] = t.timeit(number=1)


json.dump(times, open('data/timings.txt', 'w'))


