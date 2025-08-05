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

C++
***

.. rubric:: Required

- C++ compiler (supporting the C++20 standard)
- `Basix <https://github.com/FEniCS/basix>`_ (C++ interface)
- `Boost <https://www.boost.org>`_
- `CMake <https://cmake.org>`_ [build dependency]
- HDF5 (with MPI support enabled)
- MPI (MPI-3 or later).
- `pkg-config <https://www.freedesktop.org/wiki/Software/pkg-config/>`_
- `pugixml <https://pugixml.org/>`_
- `spdlog <https://github.com/gabime/spdlog/>`_
- UFCx [``ufcx.h``, provided by FFCx package or FFCx UFCx CMake install
  at ``ffcx/cmake/*``]
- At least one of ParMETIS [2]_, KaHIP or PT-SCOTCH [2]_

From ParMETIS, KaHIP or PT-SCOTCH, ParMETIS is recommended.

.. rubric:: Optional

- `ADIOS2 <https://github.com/ornladios/ADIOS2/>`_ (additional parallel
  IO support)
- `PETSc <https://petsc.org/>`_ [1]_
- `SLEPc <https://slepc.upv.es/>`_ (eigenvalue computations)

.. rubric:: Optional for demos

- FFCx

PETSc and FFCx are optional but recommended.

Python interface
****************

Requirements for the Python interface, in addition to the C++
requirements.

.. rubric:: Required

- Python
- Python CFFI (https://cffi.readthedocs.io/)
- FFCx, UFL and Basix Python interface.
- mpi4py (https://mpi4py.readthedocs.io/)
- nanobind (https://github.com/wjakob/nanobind)
- NumPy (https://www.numpy.org)
- scikit-build-core[pyproject] (https://scikit-build-core.readthedocs.io)

.. rubric:: Optional

- petsc4py (recommended)

.. rubric:: Optional for demos

- Numba
- pyamg
- pyvista (for plotting)
- slepc4py

Building and installing
^^^^^^^^^^^^^^^^^^^^^^^

C++
***

The C++ library is built using CMake. Create a build directory in
``cpp/``, e.g. ``mkdir -p build/`` and in the build run directory::

    cmake ../
    make install

To set the installation prefix::

    cmake -DCMAKE_INSTALL_PREFIX=<my-install-path> ../
    make install


Python
******

After installation of the C++ interface, from the ``python/`` directory
the Python interface can be installed using::

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
