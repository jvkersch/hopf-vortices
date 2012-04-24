import scipy.io as sio 


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
    sio.savemat(filename, {'X0': X0, 'gamma': gamma, 'sigma' : sigma}, 
                oned_as='column')


def load_variables(filename, varnames):
    c = sio.loadmat(filename)
    return [c[name] for name in varnames]


def save_variables(filename, vardict):
    sio.savemat(filename, vardict, oned_as='column')

