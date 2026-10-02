"""Build script for cgrbptools.

All project metadata lives in pyproject.toml. This file exists only because
setuptools cannot declare C/Cython extensions declaratively, and cgrbptools
needs to compile the Cython sources vendored in the PyLk submodule.

Nothing here modifies the submodule: the extensions are named with their full
dotted install path, so build_ext places the compiled .so files alongside their
.pyx sources in cgrbptools/evals/PyLk/pylk/, which is where pylk's relative
imports (`from ._writhemap_cython import ...`) look for them.
"""

from pathlib import Path

from setuptools import Extension, setup

HERE = Path(__file__).parent
PYLK_PKG = "cgrbptools.evals.PyLk.pylk"
PYLK_DIR = "cgrbptools/evals/PyLk/pylk"

CYTHON_MODULES = ["_writhemap_cython", "linkingnumber_cython"]


def cython_extensions():
    missing = [m for m in CYTHON_MODULES if not (HERE / PYLK_DIR / f"{m}.pyx").exists()]
    if missing:
        # PyLk submodule is not checked out (e.g. a clone without
        # --recurse-submodules). pylk degrades to its numba/python
        # implementations at runtime, so build without the extensions rather
        # than failing the install outright.
        print(
            f"cgrbptools: PyLk sources missing ({', '.join(missing)}); skipping "
            "Cython extensions. pylk will fall back to its numba implementation."
        )
        return []

    import numpy
    from Cython.Build import cythonize

    extensions = [
        Extension(
            f"{PYLK_PKG}.{module}",
            [f"{PYLK_DIR}/{module}.pyx"],
            include_dirs=[numpy.get_include()],
        )
        for module in CYTHON_MODULES
    ]

    # annotate=False must be explicit. Left unset it defaults to None, and
    # Cython then auto-enables annotation whenever a generated .html already
    # exists next to the source. PyLk tracks pylk/linkingnumber_cython.html, so
    # the default would rewrite it and leave the submodule dirty after a build.
    #
    # build_dir keeps the generated .c out of the submodule working tree. Left
    # at the default they are written next to the .pyx, where setuptools then
    # ships them inside the wheel (~2.6 MB of dead weight).
    return cythonize(
        extensions,
        language_level="3",
        annotate=False,
        build_dir="build/cython",
    )


setup(ext_modules=cython_extensions())
