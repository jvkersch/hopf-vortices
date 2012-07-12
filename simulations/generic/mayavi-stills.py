from mayavi import mlab
mlab.options.offscreen = True

from hopf.util.matlab_io import load_variables

from math import exp

import numpy as np
import numpy.linalg as LA 

# Mesh data for sphere (global)

r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2*pi:101j]

x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

# Vortex locations
times, vortices = \
    load_variables('data/generic_sphere_40', ['times', 'vortices'])
# Gamma, sigma
gamma, sigma = \
    load_variables('random_initial_conditions40.mat', ['gamma', 'sigma'])



def smeared_out_vorticity(point, v):
    omega = 0
    for k in xrange(0, v.shape[0]):
        d = LA.norm(point - v[k, :])
        omega += gamma[k]*exp(-d**2./sigma**2.)
        
    return omega

def generate_scalars(v):
    n, m = phi.shape
    s = np.empty((n, m))
    for i in xrange(0, n):
        for j in xrange(0, m):
            alpha = phi[i, j]
            beta = theta[i, j]
            px = r*sin(alpha)*cos(beta)
            py = r*sin(alpha)*sin(beta)
            pz = r*cos(alpha)
            s[i, j] = smeared_out_vorticity(np.array([px, py, pz]), v[:, :])
    s /= s.max()
    return s


def create_frame(n):

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))#, size=(400, 300))

    s = generate_scalars(vortices[n, :, :])

    sphere = mlab.mesh(x, y, z, scalars=s, resolution=50, opacity=0.7)
    sphere.actor.property.backface_culling = True

    mlab.savefig('still%d.png' % n)


if __name__ == '__main__':
    for n in xrange(0, vortices.shape[0]):
        print "%d of %d" % (n, vortices.shape[0])
        create_frame(n)
    
