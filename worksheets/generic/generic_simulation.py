from hopf.simulation import Simulation

s = Simulation()
s.load_initial_conditions('random40.mat')
s.sigma = 0.10
s.run_simulation(h = 0.1, tmax=2000, numpoints=1000, sim='sphere-midpoint')
s.post_process()
s.save_results('data/generic40_T200_sphere.mat')

s = Simulation()
s.load_initial_conditions('random40.mat')
s.sigma = 0.10
s.run_simulation(h = 0.1, tmax=2000, numpoints=1000, sim='rk4')
s.post_process()
s.save_results('data/generic40_T200_rk4.mat')

s = Simulation()
s.load_initial_conditions('random40.mat')
s.sigma = 0.10
s.run_simulation(h = 0.1, tmax=2000, numpoints=1000, sim='midpoint')
s.post_process()
s.save_results('data/generic40_T200_mp.mat')

s = Simulation()
s.load_initial_conditions('random40.mat')
s.sigma = 0.10
s.run_simulation(h = 0.1, tmax=2000, numpoints=1000, sim='lie-poisson')
s.post_process()
s.save_results('data/generic40_T200_lp.mat')



