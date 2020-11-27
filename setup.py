from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("./ctc_pyx/ctc.pyx"),
    include_dirs=[np.get_include()]
)

# не обязательно
setup(
    ext_modules=cythonize("./ctc_pyx/ctc_logspace.pyx"),
    include_dirs=[np.get_include()]
)
