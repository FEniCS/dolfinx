.. DOLFIN installation docs

============
Installation
============


Building from source
====================


Dependencies
------------

DOLFIN-X requires a compiler that supports the C++14 standard.

The required and optional DOLFIN dependencies are listed below.

Required for C++ core
^^^^^^^^^^^^^^^^^^^^^

- Boost (http://www.boost.org), with the following compiled Boost
  components

  - filesystem
  - iostreams
  - program_options
  - timer

- CMake (https://cmake.org)
- Eigen3 (http://eigen.tuxfamily.org)
- pkg-config (https://www.freedesktop.org/wiki/Software/pkg-config/)
- Python (used by the build system)
- HDF5 (with MPI support enabled)
- MPI
- ParMETIS [1]_
- PETSc (strongly recommended) [2]_
- SCOTCH and PT-SCOTCH [1]_
- SLEPc
- Suitesparse [1]_


Required for Python interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Python (including header files)
- pybind11 (https://github.com/pybind/pybind11)
- NumPy (http://www.numpy.org)


Optional for the Python interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Matplotlib (required for plotting)
- mpi4py
- petsc4py
- slepc4py


.. [1] It is strongly recommended to use the PETSc build system to
       download and configure and build these libraries.

.. [2] Its is recommended to configuration with ParMETIS, PT-SCOTCH,
       MUMPS and Hypre using the
       ``--download-parmetis --download-ptscotch --download-suitesparse
       --download-mumps --download-hypre``
