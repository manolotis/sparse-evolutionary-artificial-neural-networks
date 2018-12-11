from distutils.core import setup
from Cython.Build import cythonize
import numpy



setup(
    ext_modules = cythonize("sparseoperations.pyx")
)


# setup(
#     ext_modules = cythonize("sparseoperations.pyx", include_path=[numpy.get_include(), "./", "numpy", ",.numpy"])
# )
