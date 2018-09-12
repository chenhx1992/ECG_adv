#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 22:05:07 2018

@author: chenhx1992
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy 

ext_modules=[ Extension("softdtwc",
              ["softdtwc.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"],
              include_dirs=[numpy.get_include()])]

setup(
  name = "softdtwc",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

#setup(
#    ext_modules=cythonize("softdtwc.pyx"),
#    include_dirs=[numpy.get_include()]
#)


#import numpy
#
#def configuration(parent_package=None, top_path=None):
#    from numpy.distutils.misc_util import Configuration
#    
#    config = Configuration('softdtwc', parent_package, top_path)
#    
#    config.add_extension('softdtwc', sources=['softdtwc.c'], include_dirs=[numpy.get_include()])
#    
#    return config
#
#if __name__ == '__main__':
#    from numpy.distutils.core import setup
#    setup(**configuration(top_path='').todict())

#test 
#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(
#    ext_modules=cythonize("fib.pyx"),
#)