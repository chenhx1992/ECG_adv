#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:35:11 2018

@author: chenhx1992
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy 

ext_modules=[ Extension(
                "softdtwc_wd",
                ["softdtwc_wd.pyx"],
                libraries=["m"],
                extra_compile_args = ['-ffast-math', '-fopenmp'],
                extra_link_args=['-fopenmp'],
                include_dirs=[numpy.get_include()],
              )
]

setup(
  name = "softdtwc_wd",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)