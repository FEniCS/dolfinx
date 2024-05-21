Finite element (``dolfinx::fem``)
=================================

*Under development*

Finite elements
---------------

.. doxygenclass:: dolfinx::fem::FiniteElement
   :project: DOLFINx
   :members:


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

Assembly
--------

.. doxygenfile:: fem/assembler.h
   :project: DOLFINx
   :sections: func


Interpolation
-------------

.. doxygenfunction:: dolfinx::fem::interpolation_coords
   :project: DOLFINx


Sparsity pattern construction
-----------------------------

.. doxygenfunction:: dolfinx::fem::create_sparsity_pattern(const Form<T, U>&)
   :project: DOLFINx


PETSc helpers
-------------

.. doxygennamespace:: dolfinx::fem::petsc
   :project: DOLFINx
   :content-only:


Misc
----

.. doxygenfile:: fem/utils.h
   :project: DOLFINx
   :no-link:
   :sections: func
..    :path: ../../../cpp/dolfinx/fem/

.. .. .. doxygennamespace:: dolfinx::fem
.. ..    :project: DOLFINx
.. ..    :members:
