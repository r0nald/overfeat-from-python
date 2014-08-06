'''
    You must first do make all in OverFeat/src
'''
from distutils.core import setup, Extension
import numpy
import os.path as path

module1 = Extension("overfeatfunctions",
                    include_dirs = ['../OverFeat/src',
                                    '../OverFeat/data/default',
                                    numpy.get_include()],
                    library_dirs = ['../OverFeat/src'],
                    libraries = ['openblas'],
                    sources = ['overfeatfunctions.cpp',
                               '../OverFeat/src/overfeat.cpp',
                               '../OverFeat/API/python/overfeatmodule.cpp'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-lgomp',
                    '/home/ronald/src/overfeat-from-python/OverFeat/src/libTH.a'])

setup(name = 'overfeat',
      version = '1.0',
      description = 'Python bindings for overfeat',
      ext_modules = [module1],
      install_requires = ['numpy'])
