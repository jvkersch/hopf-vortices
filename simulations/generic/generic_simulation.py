from hopf.simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('random_initial_conditions.mat')
s.run_simulation(h=.1, sim='sphere', tmax=1000)
s.post_process()
s.save_results('data/random_h01.mat')

s.run_simulation(h=.01, sim='sphere', tmax=1000)
s.post_process()
s.save_results('data/random_h001.mat')
