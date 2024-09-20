.. DOLFINx installation docs

Installation
============

DOLFINx can be installed using various packages managers, run using
containers, or built manually from source.

Binaries
--------

See the `README.md <https://github.com/FEniCS/dolfinx/blob/main/README.md#installation>`_
for recommendations and instructions.

Source
------

Installation of DOLFINx requires installation of the C++ core. Most
users will also want the Python interface.

Dependencies
^^^^^^^^^^^^

C++ core
********

.. rubric:: Required

- C++ compiler (supporting the C++20 standard)
- `Basix <https://github.com/FEniCS/basix>`_
- `Boost <https://www.boost.org>`_, with the following compiled Boost
  components

  - timer

- `CMake <https://cmake.org>`_ [build dependency]
- HDF5 (with MPI support enabled)
- MPI supporting MPI version 3 or above.
- `pkg-config <https://www.freedesktop.org/wiki/Software/pkg-config/>`_
- `pugixml <https://pugixml.org/>`_
- `spdlog <https://github.com/gabime/spdlog/>`_
- UFCx [``ufcx.h``, provided by FFCx]
- At least one of ParMETIS [2]_, KaHIP or PT-SCOTCH [2]_

From ParMETIS, KaHIP or PT-SCOTCH, ParMETIS is recommended.

.. rubric:: Optional

- `ADIOS2 <https://github.com/ornladios/ADIOS2/>`_ (additional parallel
  IO support)
- `PETSc <https://petsc.org/>`_ [1]_
- `SLEPc <https://slepc.upv.es/>`_ (eigenvalue computations)

PETSc is optional but still recommended.

Python interface
****************

Below are additional requirements for the Python interface.

.. rubric:: Required

- Python
- FFCx, UFL and Basix (https://github.com/FEniCS/).
- mpi4py
- nanobind (https://github.com/wjakob/nanobind)
- NumPy (https://www.numpy.org)
- scikit-build-core[pyproject] (https://scikit-build-core.readthedocs.io)

.. rubric:: Optional

- Numba
- petsc4py (recommended)
- pyamg
- pyvista (for plotting)
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

    pip install -r build-requirements.txt
    pip install --check-build-dependencies --no-build-isolation .


.. rubric:: Footnotes

.. [1] Its is recommended to configure with ParMETIS, PT-SCOTCH,
       MUMPS and Hypre using
       ``--download-parmetis --download-ptscotch --download-suitesparse
       --download-mumps --download-hypre``. macOS users should 
       additionally configure MUMPS via PETSc with 
       ``--download-mumps-avoid-mpi-in-place``.

.. [2] PETSc can download and configure and build these libraries.
