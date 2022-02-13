Finite element (``dolfinx::fem``)
=================================

All
---

.. doxygennamespace:: dolfinx::fem
   :project: DOLFINx
   :content-only:

Finite elements
---------------

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

.. doxygenfile:: fem/assembler.h
   :project: DOLFINx


Dirichlet boundary conditions
-----------------------------


Assembly
--------

.. doxygenfile:: fem/assembler.h
   :project: DOLFINx


Interpolation
-------------


Sparsity pattern construction
-----------------------------



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

.. doxygenclass:: dolfinx::fem::DofMap
   :project: DOLFINx
   :members:


Degree-of-freedom maps 2
------------------------

.. doxygenfile:: fem/DofMap.h
   :project: DOLFINx
   :sections: briefdescription


Foo
---

.. doxygenfile:: fem/utils.h
   :project: DOLFINx
   :no-link:
   :sections: func

.. .. doxygennamespace:: dolfinx::fem
..    :project: DOLFINx
..    :members:

