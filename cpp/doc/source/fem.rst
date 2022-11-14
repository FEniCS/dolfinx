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

.. doxygenfunction:: dolfinx::fem::interpolate(Function<T> &u, const Function<T> &v, std::span<const std::int32_t> cells)
   :project: DOLFINx


.. doxygenfunction:: dolfinx::fem::interpolate(Function<T>& u, std::span<const T> f, std::array<std::size_t, 2> fshape, std::span<const std::int32_t> cells)
   :project: DOLFINx


Sparsity pattern construction
-----------------------------

.. doxygenfunction:: dolfinx::fem::create_sparsity_pattern(const Form<T>&)
   :project: DOLFINx

.. doxygenfunction:: dolfinx::fem::create_sparsity_pattern(const mesh::Topology &topology, const std::array<std::reference_wrapper<const DofMap>, 2> &dofmaps, const std::set<IntegralType> &integrals)
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
