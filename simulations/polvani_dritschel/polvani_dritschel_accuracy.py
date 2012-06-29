
import timeit
from hopf.util.matlab_io import save_variables
from numpy import logspace



setup_stmt = """
from hopf.simulation import Simulation
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

"""

from hopf.simulation import Simulation
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')


for integrator in ['sphere', 'midpoint', 'lie-poisson']:

    for n, h in enumerate(logspace(-3, -1, 10)):
        # Statement to time (run simulation but do not save results)
        stmt="s.run_simulation(h=%f, tmax=20, sim=\'%s\')" % (h, integrator)
        t = timeit.Timer(stmt=stmt, setup=setup_stmt)
        
        time = t.timeit(number=1)/1

        s.run_simulation(h=h, tmax=20, sim=integrator)

        # Save output in new file
        filename = 'data/pd_timed_%s_%d.mat' % (integrator, n)
        save_variables(filename, {'h': h, 'time': time, 
                                  'vortices': s.vortices, 'times': s.times})


    



