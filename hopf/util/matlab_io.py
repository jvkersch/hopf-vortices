import scipy.io as sio 
import os.path

def load_initial_conditions(filename):
    c = sio.loadmat(filename)

    sigma = 0
    if 'sigma' in c:
        sigma = c['sigma']

    X0 = c['X0']
    gamma = c['gamma'].reshape(X0.shape[0])

    sigma = sigma[0][0]
    return gamma, X0, sigma


def save_initial_conditions(filename, gamma, X0, sigma=0.0):

    vardict = {'X0': X0, 'gamma': gamma, 'sigma' : sigma}
    save_variables(filename, vardict)


def load_variables(filename, varnames):
    c = sio.loadmat(filename)
    old_vars = [c[name] for name in varnames]
    new_vars = []
    for var in old_vars: # Remove spurious dimensions
        new_shape = tuple(t for t in var.shape if t != 1)
        var.shape = new_shape
        new_vars.append(var)
    return new_vars


def save_variables(filename, vardict):
    # Create directory if it doesn't exist
    base = os.path.dirname(filename)
    if len(base) > 0 and not os.path.exists(base):
        os.makedirs(base)

    sio.savemat(filename, vardict, oned_as='column')

