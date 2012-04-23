import scipy.io as sio 


def load_ic_from_matfile(filename):
    c = sio.loadmat(filename)

    sigma = c['sigma']
    X0 = c['X0']
    gamma = c['gamma'].reshape(X0.shape[1])

    sigma = sigma[0][0]
    return gamma, X0, sigma


def save_ic_to_matfile(filename, gamma, X0):
    sio.savemat(filename, {'X0': X0, 'gamma': gamma}, oned_as='column')


def load_variables_from_matfile(filename, varnames):
    c = sio.loadmat(filename)
    return [c[name] for name in varnames]


def save_variables_to_matfile(filename, vardict):
    sio.savemat(filename, vardict, oned_as='column')

