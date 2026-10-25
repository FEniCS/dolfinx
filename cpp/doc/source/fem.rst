Finite element (``dolfinx::fem``)
=================================

Finite elements
---------------

.. doxygenclass:: dolfinx::fem::FiniteElement
   :project: DOLFINx
   :members:

.. doxygenenum:: dolfinx::fem::doftransform
   :project: DOLFINx


Function spaces and functions
-----------------------------

Finite element functions, expressions and constants

Function spaces
^^^^^^^^^^^^^^^

.. doxygenclass:: dolfinx::fem::FunctionSpace
   :project: DOLFINx
   :members:

Functions
^^^^^^^^^

.. doxygenclass:: dolfinx::fem::Function
   :project: DOLFINx
   :members:


Constants
^^^^^^^^^

.. doxygenclass:: dolfinx::fem::Constant
   :project: DOLFINx
   :members:


Forms
-----

.. doxygenclass:: dolfinx::fem::Form
   :project: DOLFINx
   :members:

.. doxygenenum:: dolfinx::fem::IntegralType
   :project: DOLFINx


Dirichlet boundary conditions
-----------------------------

.. doxygenclass:: dolfinx::fem::DirichletBC
   :project: DOLFINx
   :members:


Degree-of-freedom maps
----------------------

.. doxygenfunction:: dolfinx::fem::transpose_dofmap
   :project: DOLFINx

.. doxygenclass:: dolfinx::fem::DofMap
   :project: DOLFINx
   :members:

.. doxygenclass:: dolfinx::fem::ElementDofLayout
   :project: DOLFINx
   :members:

Assembly
--------

.. cpp:namespace-push:: dolfinx::fem

.. doxygenfile:: assembler.h
   :project: DOLFINx
   :sections: func

.. cpp:namespace-pop::


Interpolation
-------------

.. doxygenfunction:: dolfinx::fem::interpolation_coords
   :project: DOLFINx

.. cpp:namespace-push:: dolfinx::fem

.. doxygenfile:: interpolate_geometry.h
   :project: DOLFINx
   :sections: func

.. cpp:namespace-pop::


Sparsity pattern construction
-----------------------------

.. cpp:namespace-push:: dolfinx::fem

.. doxygenfile:: sparsitypattern.h
   :project: DOLFINx
   :sections: func

.. cpp:namespace-pop::


PETSc helpers
-------------

.. doxygennamespace:: dolfinx::fem::petsc
   :project: DOLFINx
   :content-only:


Creating forms, function spaces and expressions
-----------------------------------------------

.. cpp:namespace-push:: dolfinx::fem

.. doxygenfile:: form_factory.h
   :project: DOLFINx
   :sections: func

.. cpp:namespace-pop::

.. cpp:namespace-push:: dolfinx::fem

.. doxygenfile:: functionspace_factory.h
   :project: DOLFINx
   :sections: func

.. cpp:namespace-pop::

.. cpp:namespace-push:: dolfinx::fem

.. doxygenfile:: expression_factory.h
   :project: DOLFINx
   :sections: func

.. cpp:namespace-pop::


Integration domains
-------------------

.. cpp:namespace-push:: dolfinx::fem

.. doxygenfile:: integration_domains.h
   :project: DOLFINx
   :sections: func

.. cpp:namespace-pop::
