#!/usr/bin/env python3

from distutils.core import setup, Extension


module1 = Extension(
        'cwt1d.cwtcore',
        sources=[
            'src/pycwt1d.cpp',
            'src/fftw_blitz.cpp',
        ],
        include_dirs=['deps/include'],
        libraries=[
            'boost_python3',
            'boost_numpy3',
            'gsl',
            'gslcblas',
            'fftw3f',
            'fftw3_threads',
            'fftw3f_threads',
        ],
)

setup(name='cwtcore',
      packages=['cwt1d'],
      version='1.0',
      ext_modules=[module1])
