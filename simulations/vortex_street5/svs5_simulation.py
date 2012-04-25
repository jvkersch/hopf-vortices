from simulation import Simulation
                                       
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.run_simulation()
s.post_process()
s.save_results()
