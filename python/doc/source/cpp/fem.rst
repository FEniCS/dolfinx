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

Function space
^^^^^^^^^^^^^^

.. doxygenclass:: dolfinx::fem::FunctionSpace
   :project: DOLFINx
   :members:

Function
^^^^^^^^

.. doxygenclass:: dolfinx::fem::Function
   :project: DOLFINx
   :members:

Constant
^^^^^^^^

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

.. doxygenfile:: DirichletBC.h
   :project: DOLFINx
   :path: ../../../cpp/dolfinx/fem/
   :sections: func


.. .. doxygenclass:: dolfinx::fem::DirichletBC
..    :project: DOLFINx
..    :members:


Assembly
--------

.. doxygenfile:: fem/assembler.h
   :project: DOLFINx


Interpolation
-------------

.. doxygenfile:: dolfinx/fem/interpolation.h
   :project: DOLFINx


Sparsity pattern construction
-----------------------------

.. .. doxygenfunction:: dolfinx::fem::create_sparsity_pattern
..    :project: DOLFINx

.. doxygenfunction:: dolfinx::fem::create_sparsity_pattern(const Form<T>&)
   :project: DOLFINx

.. doxygenfunction:: dolfinx::fem::create_sparsity_pattern(const mesh::Topology &topology, const std::array<std::reference_wrapper<const DofMap>, 2> &dofmaps, const std::set<IntegralType> &integrals)
   :project: DOLFINx


PETSc helpers
-------------

.. doxygennamespace:: dolfinx::fem::petsc
   :project: DOLFINx


.. Functions and expressions
.. -------------------------

.. .. doxygenclass:: dolfinx::fem::Function
..    :project: DOLFINx
..    :members:

.. .. doxygenclass:: dolfinx::fem::Constant
..    :project: DOLFINx
..    :members:


.. Forms
.. -----

.. .. doxygenclass:: dolfinx::fem::Form
..    :project: DOLFINx
..    :members:


Degree-of-freedom maps
----------------------

.. doxygenfunction:: dolfinx::fem::transpose_dofmap
   :project: DOLFINx
   :members:

.. doxygenclass:: dolfinx::fem::DofMap
   :project: DOLFINx
   :members:


.. Degree-of-freedom maps 2
.. ------------------------

.. .. doxygenfile:: DofMap.h
..    :project: DOLFINx
..    :path: ../../../cpp/dolfinx/fem/

Foo
---

.. doxygenfile:: fem/utils.h
   :project: DOLFINx
   :no-link:
   :sections: func
..    :path: ../../../cpp/dolfinx/fem/

.. .. doxygennamespace:: dolfinx::fem
..    :project: DOLFINx
..    :members:
