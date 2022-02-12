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
   dolfinx.geometry
   dolfinx.graph
   dolfinx.io
   dolfinx.jit
   dolfinx.la
   dolfinx.mesh
   dolfinx.nls
   dolfinx.pkgconfig
   dolfinx.plot


pybind11/C++interface
^^^^^^^^^^^^^^^^^^^^^

These are low-level interfaces to the C++ component of DOLFINx. These
interfaces are subject to change and not generally intended for
application-level use.

.. autosummary::
   :toctree: generated

   dolfinx.cpp.common
   dolfinx.cpp.fem
   dolfinx.cpp.geometry
   dolfinx.cpp.graph
   dolfinx.cpp.io
   dolfinx.cpp.log
   dolfinx.cpp.mesh
   dolfinx.nls
   dolfinx.cpp.refinement


C++
===

The C++ API is documented `here
<https://docs.fenicsproject.org/dolfinx/main/cpp/>`_.
