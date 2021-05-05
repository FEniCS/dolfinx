.. DOLFINx installation docs

============
Installation
============

Installation of DOLFINx requires installation of the C++ core. Most
users will also want the Python interface.

Dependencies
============

C++ core
--------

.. rubric:: Required

- C++ compiler (supporting the C++17 standard)
- Boost (http://www.boost.org), with the following compiled Boost
  components

  - filesystem
  - timer

- CMake (https://cmake.org) [build dependency]
- xtensor (https://xtensor.readthedocs.io/) and xtensor-blas (https://xtensor-blas.readthedocs.io/)
- pkg-config (https://www.freedesktop.org/wiki/Software/pkg-config/)
- Python 3 [build dependency]
- FFCx [build dependency, for ``ufc.h`` and ``ufc_geometry.h`` headers]
- MPI
- HDF5 (with MPI support enabled)
- PETSc [2]_
- SCOTCH and PT-SCOTCH [1]_  (required for parallel mesh computation)

.. rubric:: Optional

- KaHIP
- ParMETIS [1]_
- SLEPc


Python interface
----------------

Below are additional requirements for the Python interface.

.. rubric:: Required

- Python
- FFCx, UFL and Basix (http://github.com/FEniCS/).
- pybind11 (https://github.com/pybind/pybind11)
- NumPy (http://www.numpy.org)
- mpi4py
- petsc4py


.. rubric:: Suggested

- pyvista (required for plotting)
- Numba
- slepc4py


Building and installing
=======================

C++ core
--------

The C++ core is built using CMake. Create a build directory in ``cpp/``,
e.g. ``mkdir -p build/`` and in the build run directory::

    cmake ../
    make install

To set the installation prefix::

    cmake -DCMAKE_INSTALL_PREFIX=<my-install-path> ../
    make install


Python
------

After installation of the C++ core, from the ``python/`` directory the
Python interface can be installed using::

    pip3 install .


Docker container
================

A Docker container is available at
https://hub.docker.com/r/dolfinx/dolfinx. The `Dockerfile
<https://github.com/FEniCS/dolfinx/blob/master/Dockerfile>`_
provides a definitive build recipe.


.. rubric:: Footnotes

.. [1] It is strongly recommended to use the PETSc build system to
       download and configure and build these libraries.

.. [2] Its is recommended to configure with ParMETIS, PT-SCOTCH,
       MUMPS and Hypre using the
       ``--download-parmetis --download-ptscotch --download-suitesparse
       --download-mumps --download-hypre``
