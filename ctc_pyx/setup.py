from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("ctc.pyx", annotate=True),
    include_dirs=[np.get_include()]
)

setup(
    ext_modules=cythonize("ctc_logspace.pyx", annotate=True),
    include_dirs=[np.get_include()]
)
