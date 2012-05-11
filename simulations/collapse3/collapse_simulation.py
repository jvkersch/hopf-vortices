from hopf.simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('collapse3.mat')

# No regularization
s.sigma = 0.0
s.run_simulation(tmax=15, numpoints=75, sim='sphere')
s.post_process()
s.save_results('data/collapse3_sphere_sigma00_sim.mat')

# Medium regularization
s.sigma = 0.10
s.run_simulation(tmax=15, numpoints=75, sim='sphere')
s.post_process()
s.save_results('data/collapse3_sphere_sigma10_sim.mat')

# Large regularization
s.sigma = 0.25
s.run_simulation(tmax=15, numpoints=75, sim='sphere')
s.post_process()
s.save_results('data/collapse3_sphere_sigma25_sim.mat')

# Runge-Kutta, medium regularization
s.sigma = 0.10
s.run_simulation(tmax=15, numpoints=75, sim='rk4')
s.post_process()
s.save_results('data/collapse3_rk4_sigma10_sim.mat')

# Output for plotting 
s.sigma = 0.10
s.run_simulation(tmax=10, numpoints=1000, h=0.01, sim='sphere')
s.post_process()
s.save_results('data/collapse3_sphere_sigma10_hi_sim.mat')
