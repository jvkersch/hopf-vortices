from distutils.core import setup 
from distutils.extension import Extension 
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("vectorized_so3", 
                         ["vectorized_so3.pyx"], 
                         include_dirs=[numpy.get_include()])]
  
setup( 
    name = 'Lie-Poisson integrator', 
    cmdclass = {'build_ext': build_ext}, 
    ext_modules = ext_modules
)
