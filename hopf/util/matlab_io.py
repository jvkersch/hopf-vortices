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
    return [c[name] for name in varnames]


def save_variables(filename, vardict):
    # Create directory if it doesn't exist
    base = os.path.dirname(filename)
    if not os.path.exists(base):
        os.makedirs(base)

    sio.savemat(filename, vardict, oned_as='column')

