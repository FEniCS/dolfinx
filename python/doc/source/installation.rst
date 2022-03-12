.. DOLFINx installation docs

Installation
============

DOLFINx can be installed using various packages managers, run using
containers, or built manually from source.

`Spack <https://spack.io/>`_ is the recommended installation tool for
high performance computers


Spack
-----

See `README.md <https://github.com/FEniCS/dolfinx/blob/main/README.md>`_
for instructions.


Docker
------

A nightly Docker container is provided at
https://hub.docker.com/r/dolfinx/dolfinx. The `Dockerfile
<https://github.com/FEniCS/dolfinx/blob/master/Dockerfile>`_ provides a
definitive build recipe. See `README.md
<https://github.com/FEniCS/dolfinx/blob/main/README.md>`_ for
instructions.

Versioned Docker images will be provided in the future.


Debian/Ubuntu packages
----------------------

The Debian/Ubuntu package is ``fenics-dolfinx``. The Ubuntu PPA is at
https://launchpad.net/~fenics-packages/+archive/ubuntu/fenics and Debian
packages are at
https://packages.debian.org/search?keywords=python3%2Ddolfinx.


Conda
-----

*Under development*

pip
---

*Under development*


Source
------

Installation of DOLFINx requires installation of the C++ core. Most
users will also want the Python interface.


Dependencies
^^^^^^^^^^^^

C++ core
********

.. rubric:: Required

- C++ compiler (supporting the C++17 standard)
- `Boost <http://www.boost.org>`_, with the following compiled Boost
  components

  - timer

- `CMake <https://cmake.org>`_ [build dependency]
- `xtensor <https://xtensor.readthedocs.io/>`_
- `pkg-config <https://www.freedesktop.org/wiki/Software/pkg-config/>`_
- `Basix <http://github.com/FEniCS/basix>`_
- UFCx [``ufcx.h``, provided by FFCx]
- MPI
- HDF5 (with MPI support enabled)
- `PETSc <https://petsc.org/>`_ [1]_
- At least one of ParMETIS [2]_, KaHIP or PT-SCOTCH [2]_

From ParMETIS, KaHIP or PT-SCOTCH, ParMETIS is recommended.

.. rubric:: Optional

- `ADIOS2 <https://github.com/ornladios/ADIOS2/>`_ (additional parallel
  IO support)
- `SLEPc <https://slepc.upv.es/>`_ (eigenvalue computations)


Python interface
****************

Below are additional requirements for the Python interface.

.. rubric:: Required

- Python
- FFCx, UFL and Basix (http://github.com/FEniCS/).
- pybind11 (https://github.com/pybind/pybind11)
- NumPy (http://www.numpy.org)
- mpi4py
- petsc4py

.. rubric:: Suggested

- pyvista (for plotting)
- Numba
- slepc4py


Building and installing
^^^^^^^^^^^^^^^^^^^^^^^

C++ core
********

The C++ core is built using CMake. Create a build directory in ``cpp/``,
e.g. ``mkdir -p build/`` and in the build run directory::

    cmake ../
    make install

To set the installation prefix::

    cmake -DCMAKE_INSTALL_PREFIX=<my-install-path> ../
    make install


Python
******

After installation of the C++ core, from the ``python/`` directory the
Python interface can be installed using::

    pip install .


.. rubric:: Footnotes

.. [1] Its is recommended to configure with ParMETIS, PT-SCOTCH,
       MUMPS and Hypre using the
       ``--download-parmetis --download-ptscotch --download-suitesparse
       --download-mumps --download-hypre``

.. [2] PETSc can download and configure and build these libraries.

