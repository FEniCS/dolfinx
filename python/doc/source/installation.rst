.. DOLFIN installation docs

============
Installation
============

Installation of DOLFIN-X requires installation of the C++ core. Most
users will also want the Python interface.

Dependencies
============

C++ core
--------

.. rubric:: Required

- C++ compiler (supporting the C++17 standard)
- Boost (http://www.boost.org), with the following compiled Boost components

  - filesystem
  - timer

- CMake (https://cmake.org) [build dependency]
- Eigen3 (http://eigen.tuxfamily.org)
- pkg-config (https://www.freedesktop.org/wiki/Software/pkg-config/)
- Python 3 [build dependency]
- FFC-X [build dependency, for ``ufc.h`` and ``ufc_geometry.h`` headers]
- MPI
- HDF5 (with MPI support enabled)
- PETSc [2]_

.. rubric:: Optional

- ParMETIS [1]_
- SCOTCH and PT-SCOTCH [1]_  (required for parallel mesh computation)
- SLEPc


Python interface
----------------

Below are additional requirements for the Python interface.

.. rubric:: Required

- Python 3
- FFC-X, UFL and FIAT (http://github.com/FEniCS/).
- pybind11 (https://github.com/pybind/pybind11)
- NumPy (http://www.numpy.org)
- mpi4py
- petsc4py


.. rubric:: Recommended

- Matplotlib (required for plotting)
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

    cmake -DCMAKE_INSTALL_PATH=<my-install-path> ../
    make install


Python
------

After installtion of the C++ core, from the ``python/`` directory the
Python interface can be installed using::

    pip3 install .


Docker container
================

A Docker container is available at
https://quay.io/repository/fenicsproject/dolfinx. The `Dockerfile
<https://github.com/FEniCS/dolfinx/blob/master/Dockerfile>`_
provides a definitive build recipe.


.. rubric:: Footnotes

.. [1] It is strongly recommended to use the PETSc build system to
       download and configure and build these libraries.

.. [2] Its is recommended to configuration with ParMETIS, PT-SCOTCH,
       MUMPS and Hypre using the
       ``--download-parmetis --download-ptscotch --download-suitesparse
       --download-mumps --download-hypre``
