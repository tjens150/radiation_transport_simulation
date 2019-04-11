from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name='Monte Carlo scatter sim',
    ext_modules=[Extension('_pulla_cy', ['pulla.pyx'],)],
    cmdclass={'build_ext': build_ext},
    )
   
