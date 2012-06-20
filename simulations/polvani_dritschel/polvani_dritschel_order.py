from hopf.simulation import Simulation
from numpy import logspace
import scipy

## Simulations for different time scales (sphere integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-4, -1, 10)):
    s.run_simulation(tmax=50, h=h, sim='sphere')
    s.post_process()

    filename = 'data/pd_sphere_order_%d.mat' % n
    s.save_results(filename)


## Simulations for different time scales (Lie-Poisson integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-4, -1, 10)):
    try:
	if n == 0: continue

        s.run_simulation(tmax=50, h=h, sim='lie-poisson')
        s.post_process()

        filename = 'data/pd_lp_order_%d.mat' % n
        s.save_results(filename)
    except scipy.optimize.nonlin.NoConvergence:
        print "No convergence for h = %f.\n" % h
        continue


## Simulations for different time scales (Projected midpoint integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-4, -1, 10)):
    s.run_simulation(tmax=50, h=h, sim='midpoint')
    s.post_process()

    filename = 'data/pd_mp_order_%d.mat' % n
    s.save_results(filename)
