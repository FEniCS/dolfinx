.. Documentation for the Neumann-Poisson demo from DOLFIN.

.. _demo_pde_neumann-poisson_python_documentation:

Poisson equation with pure Neumann boundary conditions
======================================================

This demo is implemented in a single Python file,
:download:`demo_neumann-poisson.py`, which contains both the
variational form and the solver.

.. include:: ../common.txt

Implementation
--------------

This description goes through the implementation in
:download:`demo_neumann-poisson.py` of a solver for the above
described Poisson equation step-by-step.

First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

    from dolfin import *

We proceed by defining a mesh of the domain.  We use a built-in mesh
provided by the class :py:class:`UnitSquareMesh
<dolfin.cpp.UnitSquareMesh>`.  In order to create a mesh consisting of
:math:`64 \times 64` squares with each square divided into two
triangles, we do as follows:

.. code-block:: python

    # Create mesh
    mesh = UnitSquareMesh(64, 64)

Next, we need to define the function space.

.. code-block:: python

    # Build function space with Lagrange multiplier
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, P1 * R)

The second argument to :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` specifies underlying
finite element, here mixed element obtained by ``*`` operator.

Now, we want to define the variational problem, but first we need to
specify the trial functions (the unknowns) and the test functions.
This can be done using
:py:class:`TrialFunctions<dolfin.functions.function.TrialFunction>`
and :py:class:`TestFunctions
<dolfin.functions.function.TestFunction>`.  It only remains to define
the source function :math:`f`, before we define the bilinear and
linear forms.  It is given by a simple mathematical formula, and can
easily be declared using the :py:class:`Expression
<dolfin.functions.expression.Expression>` class.  Note that the string
defining ``f`` uses C++ syntax since, for efficiency, DOLFIN will
generate and compile C++ code for these expressions at run-time.  The
following code shows how this is done and defines the variational
problem:

.. code-block:: python

    # Define variational problem
    (u, c) = TrialFunction(W)
    (v, d) = TestFunctions(W)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("-sin(5*x[0])", degree=2)
    a = (inner(grad(u), grad(v)) + c*v + u*d)*dx
    L = f*v*dx + g*v*ds

Since we have natural (Neumann) boundary conditions in this problem,
we don´t have to implement boundary conditions.  This is because
Neumann boundary conditions are default in DOLFIN.

To compute the solution we use the bilinear form, the linear forms,
and the boundary condition, but we also need to create a
:py:class:`Function <dolfin.functions.function.Function>` to store the
solution(s).  The (full) solution will be stored in ``w``, which we
initialize using the
:py:class:`FunctionSpace<dolfin.functions.functionspace.FunctionSpace>`
``W``.  The actual computation is performed by calling
:py:func:`solve<dolfin.fem.solving.solve>`.  The separate components
``u`` and ``c`` of the solution can be extracted by calling the split
function.  Finally, we plot the solutions to examine the result.

.. code-block:: python

    # Compute solution
    w = Function(W)
    solve(a == L, w)
    (u, c) = w.split()

    # Plot solution
    plot(u, interactive=True)

Complete code
-------------

.. literalinclude:: demo_neumann-poisson.py
   :start-after: # Begin demo
