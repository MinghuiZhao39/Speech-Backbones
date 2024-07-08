""" from https://github.com/jaywalnut310/glow-tts """

# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#     name = 'monotonic_align',
#     ext_modules = cythonize("core.pyx"),
#     include_dirs=[numpy.get_include()]
# )
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('core', ['core.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))