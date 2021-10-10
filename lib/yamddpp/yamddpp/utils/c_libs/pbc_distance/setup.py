from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

ext_modules = [
    Extension("pbc_distance",
              ["_pbc_distance_matrix.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']
              )
]

setup(
    name="pbc_distance",
    cmdclass={"build_ext": build_ext},
    include_dirs=[np.get_include()],
    ext_modules=ext_modules
)
