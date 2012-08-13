from hopf.simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('random-50-posneg-strongweak.mat')
s.run_simulation(h=.001, sim='sphere', tmax=50, numpoints=10000)
s.post_process()
s.save_results('data/random50posneg-sphere.mat')

# Midpoint integrator
s = Simulation()
s.load_initial_conditions('random-50-posneg-strongweak.mat')
s.run_simulation(h=.001, sim='midpoint', tmax=50, numpoints=10000)
s.post_process()
s.save_results('data/random50posneg-mp.mat')
