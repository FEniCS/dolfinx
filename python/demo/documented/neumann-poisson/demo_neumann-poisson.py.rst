
.. _demo_pde_neumann-poisson_python_documentation:

Poisson equation with pure Neumann boundary conditions
======================================================

This demo is implemented in a single Python file,
:download:`demo_neumann-poisson.py`, which contains both the
variational form and the solver.

This demo illustrates how to:

* Solve a linear partial differential equation with Neumann boundary
  conditions
* Use mixed finite element spaces

The solution for  :math:`u` in this demo will look as follows:

.. image:: neumann-poisson_u.png
    :scale: 75 %


Equation and problem definition
-------------------------------

The Poisson equation is the canonical elliptic partial differential
equation. For a domain :math:`\Omega \subset \mathbb{R}^n` with
boundary :math:`\partial \Omega`, the Poisson equation with particular
boundary conditions reads:

.. math::

	- \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
      \nabla u \cdot n &= g \quad {\rm on} \ \partial \Omega.

Here, :math:`f` and :math:`g` are input data and :math:`n` denotes the
outward directed boundary normal. Since only Neumann conditions are
applied, :math:`u` is only determined up to a constant :math:`c` by
the above equations. An additional constraint is thus required, for
instance:

.. math::

	\int_{\Omega} u \, {\rm d} x = 0.

This can be accomplished by introducing the constant :math:`c` as an
additional unknown (to be sought in :math:`\mathbb{R}`) and the above
constraint expressed via a Lagrange multiplier.

We further note that a necessary condition for the existence of a
solution to the Neumann problem is that the right-hand side :math:`f`
satisfies

.. math::

	\int_{\Omega} f \, {\rm d} x = - \int_{\partial\Omega} g \, {\rm d} s.

This can be seen by multiplying by :math:`1` and integrating by
parts:

.. math::

	\int_{\Omega} f \, {\rm d} x = - \int_{\Omega} 1 \cdot \nabla^{2} u \, {\rm d} x
                                     = - \int_{\partial\Omega} 1 \cdot \partial_n u \, {\rm d} s
                                       + \int_{\Omega} \nabla 1 \cdot \nabla u \, {\rm d} x
                                     = - \int_{\partial\Omega} g \, {\rm d} s.

This condition is not satisfied by the specific right-hand side chosen
for this test problem, which means that the partial differential
equation is not well-posed. However, the variational problem expressed
below is well-posed as the Lagrange multiplier introduced to satisfy
the condition :math:`\int_{\Omega} u \, {\rm d} x = 0` *effectively
redefines the right-hand side such that it safisfies the necessary
condition* :math:`\int_{\Omega} f \, {\rm d} x = -
\int_{\partial\Omega} g \, {\rm d} s`.

Our variational form reads: Find :math:`(u, c) \in V \times R` such
that

.. math::


	a((u, c), (v, d)) = L((v, d)) \quad \forall \ (v, d) \in V \times R,



.. math::

	a((u, c), (v, d)) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x
						+ \int_{\Omega} cv \, {\rm d} x
						+ \int_{\Omega} ud \, {\rm d} x, \\
	L(v)    &= \int_{\Omega} f v \, {\rm d} x
    	     	+ \int_{\partial\Omega} g v \, {\rm d} s.

:math:`V` is a suitable function space containing :math:`u` and `v`,
:math:and :math:`R` is the space of real numbers.

The expression :math:`a(\cdot, \cdot)` is the bilinear form and
:math:`L(\cdot)` is the linear form.

Note that the above variational problem may alternatively be expressed
in terms of the modified (and consistent) right-hand side
:math:`\tilde{f} = f - c`.

In this demo we shall consider the following definitions of the domain
and input functions:

* :math:`\Omega = [0, 1] \times [0, 1]` (a unit square)
* :math:`g = - \sin(5x)` (normal derivative)
* :math:`f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)` (source term)


Implementation
--------------

This description goes through the implementation in
:download:`demo_neumann-poisson.py` of a solver for the above
described Poisson equation step-by-step.

First, the :py:mod:`dolfin` module is imported: ::

    from dolfin import *

We proceed by defining a mesh of the domain.  We use a built-in mesh
provided by the class :py:class:`UnitSquareMesh
<dolfin.cpp.UnitSquareMesh>`.  In order to create a mesh consisting of
:math:`64 \times 64` squares, we do as follows: ::

    # Create mesh
    mesh = UnitSquareMesh.create(64, 64, CellType.Type.quadrilateral)

Next, we need to define the function space. ::

    # Build function space with Lagrange multiplier
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, P1 * R)

The second argument to :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` specifies underlying
finite element, here a mixed element is obtained by ``*`` operator.

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
problem: ::

    # Define variational problem
    (u, c) = TrialFunction(W)
    (v, d) = TestFunctions(W)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("-sin(5*x[0])", degree=2)
    a = (inner(grad(u), grad(v)) + c*v + u*d)*dx
    L = f*v*dx + g*v*ds

Since we have natural (Neumann) boundary conditions in this problem,
we do not have to implement boundary conditions.  This is because
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
function.  Finally, we output the solution to a ``VTK`` file to examine the result. ::

    # Compute solution
    w = Function(W)
    solve(a == L, w)
    (u, c) = w.split()

    # Save solution in VTK format
    file = File("neumann_poisson.pvd")
    file << u
