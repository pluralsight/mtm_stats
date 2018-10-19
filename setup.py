import os
import multiprocessing
from distutils.core import setup
from distutils.extension import Extension

import numpy as np

from Cython.Build import cythonize

from distutils.command.sdist import sdist as _sdist

# Read the version number
with open("mtm_stats/_version.py") as f:
    exec(f.read())

extensions = [Extension('mtm_stats.cy_mtm_stats',
                        ['mtm_stats/cy_mtm_stats.pyx',
                         'mtm_stats/mtm_stats_core.c'],
                        include_dirs=['src', np.get_include()],
                        extra_compile_args=['-fPIC','-fopenmp'],
                        extra_link_args=['-fopenmp'])]
nthreads = multiprocessing.cpu_count()

class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(extensions, nthreads=nthreads)
        _sdist.run(self)

setup(
    name='mtm_stats',
    version=__version__, # use the same version that's in _version.py
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['mtm_stats'],
    scripts=[],
    license='LICENSE.txt',
    description='Highly efficient set statistics about many-to-many relationships',
    long_description=open('README.md').read(),
    install_requires=['Cpyx>=0.2.2',
                      'numpy>=1.0',
                      'future>=0.16',
                      'cython>=0.2', # this might need to be newer?
                     ],
    extras_require = {
                      'visualize_test':  ["render_d3_fdg>=v0.2.1.6"],
                     },
    ext_modules=cythonize(extensions,
                          nthreads=nthreads),
    cmdclass={'sdist': sdist},
)
