from hopf.simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('random_initial_conditions25.mat')
s.sphere = .1
s.run_simulation(h=.01, sim='lie-poisson', tmax=20, diagnostics=True)
s.post_process()
s.save_results('data/xx.mat')
