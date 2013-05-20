from hopf.simulation import Simulation
from numpy import logspace
import scipy

## Simulations for different time scales (sphere integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-4, -1, 10)):
    s.run_simulation(tmax=10, h=h, sim='sphere-midpoint', diagnostics=True)
    s.post_process()

    filename = 'data/pd_sphere_order_%d.mat' % n
    s.save_results(filename)

## Simulations for different time scales (RK2 integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-4, -1, 10)):
    s.run_simulation(tmax=10, h=h, sim='rk2', diagnostics=True)
    s.post_process()

    filename = 'data/pd_rk_order_%d.mat' % n
    s.save_results(filename)


## Simulations for different time scales (Lie-Poisson integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-4, -1, 10)):
    try:
        s.run_simulation(tmax=10, h=h, sim='lie-poisson', diagnostics=True)
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
    s.run_simulation(tmax=10, h=h, sim='midpoint', diagnostics=True)
    s.post_process()

    filename = 'data/pd_mp_order_%d.mat' % n
    s.save_results(filename)
