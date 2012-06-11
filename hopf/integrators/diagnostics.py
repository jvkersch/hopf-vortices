import numpy as np

class Diagnostics:
    def __init__(self):

        self.niter = []
        self.res   = []

        self.n_current = 0
        self.res_current = 0.0

    def new_frame(self):
        self.niter.append(self.n_current)
        self.res.append(self.res_current)

        self.n_current = 0
        self.res_current = 0.0

    def callback(self, x, f):
        self.n_current += 1
        self.res_current = f


