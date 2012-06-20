import numpy as np

class BroydenDiagnostics:
    def __init__(self):

        self.number_iterations = []
        self.residues   = []

        self.n_current = 0
        self.res_current = 0.0

    def __call__(self, x, f):
        """See documentation for ``scipy.optimize.broyden1`` 
        for more information."""

        self.n_current += 1
        self.res_current = f

    def store(self):

        self.number_iterations.append(self.n_current)
        self.residues.append(self.res_current)

        self.n_current = 0
        self.res_current = -1
