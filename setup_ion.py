from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext='.pyx'
extensions = [Extension("pulla_ion_cy", ["pulla_ion"+ext])]

from Cython.Build import cythonize
extensions = cythonize(extensions,compiler_directives={'cdivision': True})


setup(
    name='Monte Carlo scatter sim',
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    )
   
