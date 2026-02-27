"""
Build helper for the `opencood.utils.box_overlaps` extension.

This repo vendors the generated C file (`box_overlaps.c`) so we can build without Cython.
Use:
  python opencood/utils/setup.py build_ext --inplace
"""

from __future__ import annotations

from setuptools import Extension, setup

import numpy as np

try:
    from Cython.Build import cythonize  # type: ignore

    ext_modules = [
        Extension(
            name="opencood.utils.box_overlaps",
            sources=["opencood/utils/box_overlaps.pyx"],
            include_dirs=[np.get_include()],
        )
    ]
    ext_modules = cythonize(ext_modules)
except Exception:
    ext_modules = [
        Extension(
            name="opencood.utils.box_overlaps",
            sources=["opencood/utils/box_overlaps.c"],
            include_dirs=[np.get_include()],
        )
    ]

setup(name="box overlaps", ext_modules=ext_modules)
