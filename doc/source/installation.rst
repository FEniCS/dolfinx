.. DOLFIN installation docs

============
Installation
============


Quick start
===========


Building from source
====================



Dependencies
------------

DOLFIN requires a compiler that supports the C++11 standard.

The required and optional DOLFIN dependencies are listed below.

Required
^^^^^^^^

- Boost (http://www.boost.org), with the following compiled Boost
  components

  - filesystem
  - iostreams
  - program_options
  - timer

- CMake (https://cmake.org)
- Eigen3 (http://eigen.tuxfamily.org)
- FFC (https://bitbucket.org/fenics-project/ffc)
- pkg-config (https://www.freedesktop.org/wiki/Software/pkg-config/)
- zlib


Required for Python interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Python (including header files)
- SWIG (http://www.swig.org)
- NumPy (http://www.numpy.org)
- ply (https://github.com/dabeaz/ply)


Optional
^^^^^^^^

- HDF5, with MPI support enabled
- MPI
- ParMETIS [1]_
- PETSc (strongly recommended) [2]_
- SCOTCH and PT-SCOTCH [1]_
- SLEPc
- Suitesparse [1]_
- Trilinos
- VTK


Optional for the Python interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- petsc4py
- slepc4py
- mpi4py
- Matplotlib

.. [1] It is strongly recommended to use the PETSc build system to
       download and configure and build these libraries.

.. [2] Its is recommended to configuration with ParMETIS, PT-SCOTCH,
       MUMPS and Hypre using the
       ``--download-parmetis --download-ptscotch --download-suitesparse
       --download-mumps --download-hypre``
