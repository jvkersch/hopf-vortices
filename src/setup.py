from distutils.core import setup 
from distutils.extension import Extension 
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension( "continuous_vortex_system", 
                         ["continuous_vortex_system.pyx"], 
                         include_dirs=[numpy.get_include()])]


setup( 
    name = 'Point Vortices on the Sphere', 
    cmdclass = {'build_ext': build_ext}, 
    ext_modules = ext_modules
)
