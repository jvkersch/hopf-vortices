from distutils.core import setup 
from distutils.extension import Extension 
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("adjoint", 
                         ["adjoint.pyx"], 
                         include_dirs=[numpy.get_include()]),
               Extension("so3_geometry", 
                         ["so3_geometry.pyx"], 
                         include_dirs=[numpy.get_include()])]
  
setup( 
    name = 'Lie algebra code', 
    cmdclass = {'build_ext': build_ext}, 
    ext_modules = ext_modules
)
