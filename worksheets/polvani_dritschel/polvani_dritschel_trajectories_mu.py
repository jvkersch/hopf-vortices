from hopf.simulation import Simulation

# Long trajectory, sphere integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=2000, h=.5, sim='sphere', numpoints=2000)
s.post_process()
s.save_results('data/pd_trajectory_2000_sphere.mat')

# Long trajectory, sphere-mu integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=2000, h=.5, sim='sphere-mu', numpoints=2000)
s.post_process()
s.save_results('data/pd_trajectory_2000_sphere_mu.mat')

