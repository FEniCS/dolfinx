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
   dolfinx.fem.form
   dolfinx.geometry
   dolfinx.graph
   dolfinx.io
   dolfinx.jit
   dolfinx.la
   dolfinx.mesh
   dolfinx.pkgconfig
   dolfinx.plot


pybind11 wrapped interface
^^^^^^^^^^^^^^^^^^^^^^^^^^

These are interfaces to the C++ component of DOLFINx, and are low-level
and subject to change. They are not intended for application-level
use.

.. autosummary::
   :toctree: generated

   dolfinx.cpp.common
   dolfinx.cpp.fem
   dolfinx.cpp.geometry
   dolfinx.cpp.graph
   dolfinx.cpp.io
   dolfinx.cpp.log
   dolfinx.cpp.mesh
   dolfinx.cpp.refinement


C++
===

The C++ API is documented `here
<https://docs.fenicsproject.org/dolfinx/main/cpp/>`_.
