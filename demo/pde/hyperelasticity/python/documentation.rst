.. Documentation for the hyperelasticity demo from DOLFIN.

.. _demos_pde_hyperelasticity_python_documentation:

Hyperelasticity
===============

This demo is implemented in a single Python file,
:download:`demo_hyperelasticity.py`, which contains both the
variational forms and the solver.

.. include:: ../common.txt

Implementation
--------------

This demo is implemented in the :download:`demo_hyperelasticity.py`
file.

First, the ``dolfin`` module is imported:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 13


The behavior of the form compiler FFC can be adjusted by prescribing
various parameters. Here, we want to use some of the optimization
features.

.. literalinclude:: demo_hyperelasticity.py
   :lines: 15-20

The first line tells the form compiler to use C++ compiler optimizations when
compiling the generated code. The remainder is a dictionary of options which
will be passed to the form compiler. It lists the optimizations strategies
that we wish the form compiler to use when generating code.

.. index:: VectorFunctionSpace

First, we need a tetrahedral mesh of the domain and a function space
on this mesh. Here, we choose to create a unit cube mesh with 17 ( =
16 + 1) vertices in each direction. On this mesh, we define a function
space of continuous piecewise linear vector polynomials (a Lagrange
vector element space):

.. literalinclude:: demo_hyperelasticity.py
   :lines: 23-24

Note that ``VectorFunctionSpace`` creates a function space of vector
fields. The dimension of the vector field (the number of components)
is assumed to be the same as the spatial dimension, unless otherwise
specified.

.. index:: compiled subdomain

The portions of the boundary on which Dirichlet boundary conditions
will be applied are now defined:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 26-28

The boundary subdomain ``left`` corresponds to the part of the
boundary on which :math:`x=0` and the boundary subdomain ``right``
corresponds to the part of the boundary on which :math:`x=1`. Note
that C++ syntax is used in the ``compile_subdomains`` function since
the function will be automatically compiled into C++ code for
efficiency. The (built-in) variable ``on_boundary`` is true for points
on the boundary of a domain, and false otherwise.

.. index:: compiled expression

The Dirichlet boundary values are defined using compiled expressions:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 30-35

For the ``Expression`` ``r``, the Python dictionary named ``defaults`` is used
to automatically set values in the function string.

The boundary subdomains and the boundary condition expressions are
collected together in two ``DirichletBC`` objects, one for each part
of the Dirichlet boundary:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 37-38

The Dirichlet (essential) boundary conditions are constraints on the
function space :math:`V`. The function space is therefore required as
an argument to ``DirichletBC``.

.. index:: TestFunction, TrialFunction, Constant

Trial and test functions, and the most recent approximate displacement
:math:`u` are defined on the finite element space :math:`V`, and
``Constants`` are declared for the body force (``B``) and traction
(``T``) terms:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 40-45

In place of ``Constant``, it is also possible to use ``as_vector``,
e.g.  ``B = as_vector( [0.0, -0.5, 0.0] )``. The advantage of
``Constant`` is that its values can be changed without requiring
re-generation and re-compilation of C++ code. On the other hand, using
``as_vector`` can eliminate some function calls during assembly.

With the functions defined, the kinematic quantities involved in the model
are defined using UFL syntax:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 47-54

Next, the material parameters are set and the strain energy density
and the total potential energy are defined, again using UFL syntax:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 56-64

Just as for the body force and traction vectors, ``Constant`` has been
used for the model parameters ``mu`` and ``lmbda`` to avoid
re-generation of C++ code when changing model parameters. Note that
``lambda`` is a reserved keyword in Python, hence the misspelling
``lmbda``.

.. index:: directional derivative; derivative, taking variations; derivative, automatic differentiation; derivative

Directional derivatives are now computed of :math:`\Pi` and :math:`L`
(see :eq:`first_variation` and :eq:`second_variation`):

.. literalinclude:: demo_hyperelasticity.py
   :lines: 66-70

The functional ``L`` and its Jacobian ``a``, together with the list of
Dirichlet boundary conditions ``[bcl, bcr]``, are used to construct an
nonlinear variational problem. This problem is then solved:

.. index:: VariationalProblem, form compiler parameters

.. literalinclude:: demo_hyperelasticity.py
   :lines: 72-74

The argument ``nonlinear = True`` indicates that the problem is
nonlinear and that a Newton solver should be used. The dictionary of
form compiler options, which were defined initially, is supplied using
``form_compiler_parameters = ffc_options``.

Finally, the solution ``u`` is saved to a file named
``displacement.pvd`` in VTK format, and the deformed mesh is plotted
to the screen:

.. literalinclude:: demo_hyperelasticity.py
   :lines: 76-

Complete code
-------------

.. literalinclude:: demo_hyperelasticity.py
   :start-after: # Begin demo
