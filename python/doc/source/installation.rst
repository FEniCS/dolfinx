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
- `pkg-config <https://www.freedesktop.org/wiki/Software/pkg-config/>`_ [build dependency]
- `pugixml <https://pugixml.org/>`_
- `spdlog <https://github.com/gabime/spdlog/>`_
- UFCx [``ufcx.h``, provided by FFCx package or FFCx UFCx CMake install
  at ``ffcx/cmake/*``]
- At least one of ParMETIS [2]_, KaHIP or PT-SCOTCH [2]_

From ParMETIS, KaHIP or PT-SCOTCH, ParMETIS is recommended.

.. rubric:: Optional

- `ADIOS2 <https://github.com/ornladios/ADIOS2/>`_ (additional parallel
  IO support)
- `PETSc <https://petsc.org/>`_ [1]_ (linear and non-linear problems)
- `SLEPc <https://slepc.upv.es/>`_ (eigenvalue problems)
- `SuperLU_DIST <https://github.com/xiaoyeli/superlu_dist/>`_ [2]_ 
   (linear problems with ``dolfinx::la::MatrixCSR``).

.. rubric:: Optional for demos

- FFCx

PETSc and FFCx are optional but recommended.

Python interface
****************

Requirements for the Python interface. Please see ``python/pyproject.toml`` for
precise specification. Below we use the `pypi <https://pypi.org>`_ names.

.. rubric:: Build system requirements

- Python
- DOLFINx C++ interface and all requirements.
- `scikit-build-core[pyproject] <https://scikit-build-core.readthedocs.io>`_
- `mpi4py <https://mpi4py.readthedocs.io/>`_
- `nanobind <https://github.com/wjakob/nanobind>`_ (static linking)
- petsc4py (recommended, optional)

.. rubric:: Required runtime dependencies

- Python
- Basix (Python interface), FFCx and UFL.
- `cffi <https://cffi.readthedocs.io/>`_
- `mpi4py <https://mpi4py.readthedocs.io/>`_
- `numpy <https://www.numpy.org>`_
- `pkg-config <https://www.freedesktop.org/wiki/Software/pkg-config/>`_

.. rubric:: Optional runtime dependencies

- petsc4py (linear and non-linear problems, recommended)
- numba (custom kernels and assemblers)
- `pyamg <https://github.com/pyamg/pyamg>`_ + scipy (serial linear problems)

.. rubric:: Optional for demos

- gmsh
- networkx 
- numba
- matplotlib
- petsc4py
- pyamg
- pyvista
- scipy
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

    python -m scikit_build_core.build requires | python -c "import sys, json; print(' '.join(json.load(sys.stdin)))" | xargs pip install
    pip install --check-build-dependencies --no-build-isolation .


.. rubric:: Footnotes

.. [1] Its is recommended to configure with ParMETIS, PT-SCOTCH,
       MUMPS and Hypre using
       ``--download-parmetis --download-ptscotch --download-suitesparse
       --download-mumps --download-hypre``. macOS users should
       additionally configure MUMPS via PETSc with
       ``--download-mumps-avoid-mpi-in-place``.

.. [2] PETSc can also download, configure and build these libraries.
