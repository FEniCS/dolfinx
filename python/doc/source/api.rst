=============
API reference
=============

Python
======

Public user interface
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   dolfinx.common
   dolfinx.fem
   dolfinx.fem.petsc
   dolfinx.geometry
   dolfinx.graph
   dolfinx.io
   dolfinx.io.gmshio
   dolfinx.jit
   dolfinx.la
   dolfinx.mesh
   dolfinx.nls.petsc
   dolfinx.pkgconfig
   dolfinx.plot


nanobind/C++interface
^^^^^^^^^^^^^^^^^^^^^

These are low-level interfaces to the C++ component of DOLFINx. These
interfaces are subject to change and not generally intended for
application-level use.

.. autosummary::
   :toctree: generated

   dolfinx.cpp.common
   dolfinx.cpp.fem
   dolfinx.cpp.fem.petsc
   dolfinx.cpp.geometry
   dolfinx.cpp.graph
   dolfinx.cpp.io
   dolfinx.cpp.la
   dolfinx.cpp.log
   dolfinx.cpp.mesh
   dolfinx.cpp.nls
   dolfinx.cpp.nls.petsc
   dolfinx.cpp.refinement
