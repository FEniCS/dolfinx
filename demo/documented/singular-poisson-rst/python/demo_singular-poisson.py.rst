.. Documentation for the singular Poisson demo from DOLFIN.

Singular Poisson
================

This demo is implemented in a single Python file,
:download:`demo_singular-poisson.py`, which contains both the
variational forms and the solver.

This demo illustrates how to:

* Solve a linear partial differential equation
* Apply non-zero Neumann boundary conditions
* Define Expressions
* Define a FunctionSpace
* Use the Krylov solver
* Solve singular problems

The solution for :math:`u` in this demo will look as follows:

.. image:: singular_u.png
   :scale: 75 %


Equation and problem definition
-------------------------------

The Poisson equation is the canonical elliptic partial differential
equation.  For a domain :math:`\Omega \in \mathbb{R}^n` with boundary
:math:`\Gamma = \partial \Omega`, the Poisson equation with pure
Neumann boundary conditions reads:

.. math::
   -\nabla^{2} u &= f \quad {\rm in} \ \Omega \\
   \nabla u \cdot n &= g \quad {\rm on} \ \Gamma \\

Since only Neumann conditions are applied, :math:`u` is only
determined up to a constant by the above equations. An addition
constraint is thus required, for instance

.. math::
   \int u \, {\rm d} x = 0

The most standard variational form of the Poisson equation reads: find
:math:`u \in V` such that

.. math::
   a(u, v) = L(v) \quad \forall \ v \in V,

where :math:`V` is a suitable function space and

.. math::
   a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
   L(v)    &= \int_{\Omega} f v \, {\rm d} x
   + \int_{\Gamma} g v \, {\rm d} s.

The expression :math:`a(u, v)` is the bilinear form and :math:`L(v)`
is the linear form.

If we make the Ansatz that :math:`u` can be expressed as a linear
combination of the basis functions of :math:`V`, and discretize the
equation, we can write our problem as a linear system:

.. math::
   AU = b,

where :math:`U` gives the coefficient for the basis functions
expressing :math:`u`.

Since we have pure Neumann boundary conditions, the matrix :math:`A`
is singular. There exists a non-trival vector :math:`e` such that

.. math::
   Ae=0.

span :math:`\{ e \}` is the null space of A. Consequently, the matrix
:math:`A` is rank deficient and the right-hand side vector :math:`b`
may fail to be in the column space of :math:`A`. We therefore need to
remove the components of :math:`b` that do not lie in the column space
to make the system solvable.

In this demo, we shall consider the following definitions of the input
functions, the domain, and the boundaries:

* :math:`\Omega = [0,1] \times [0,1]` (a unit square)
* :math:`\Gamma = \partial \Omega` (boundary)
* :math:`g = -\sin(5x)` (normal derivative)
* :math:`f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)` (source
  term)


Implementation
--------------

This description goes through the implementation (in
:download:`demo_singular-poisson.py`) of a solver for the above
described Poisson equation step-by-step.

First, the :py:mod:`dolfin` module is imported: ::

    from dolfin import *

Then, we check that dolfin is configured with the backend called
PETSc, since it provides us with a wide range of methods used by
:py:class:`KrylovSolver <dolfin.cpp.la.KrylovSolver>`. We set PETSc as
our backend for linear algebra. ::

    # Test for PETSc
    if not has_linear_algebra_backend("PETSc"):
        info("DOLFIN has not been configured with PETSc. Exiting.")
        exit()

    parameters["linear_algebra_backend"] = "PETSc"

We begin by defining a mesh of the domain and a finite element
function space :math:`V` relative to this mesh. We use a built-in mesh
provided by the class :py:class:`UnitSquareMesh
<dolfin.cpp.mesh.UnitSquareMesh>`. In order to create a mesh
consisting of :math:`64 \times 64` squares with each square divided
into two triangles, we do as follows: ::

    # Create mesh and define function space
    mesh = UnitSquareMesh(64, 64)
    V = FunctionSpace(mesh, "CG", 1)

Now, we need to specify the trial functions (the unknowns) and the
test functions on the space :math:`V`. This can be done using a
:py:class:`TrialFunction <dolfin.functions.function.TrialFunction>`
and a :py:class:`TestFunction
<dolfin.functions.function.TrialFunction>` as follows: ::

    u = TrialFunction(V)
    v = TestFunction(V)

Further, the source :math:`f` and the boundary normal derivative
:math:`g` are involved in the variational forms, and hence we must
specify these. Both :math:`f` and :math:`g` are given by simple
mathematical formulas, and can be easily declared using the
:py:class:`Expression <dolfin.functions.expression.Expression>`
class. Note that the strings defining f and g use C++ syntax since,
for efficiency, DOLFIN will generate and compile C++ code for these
expressions at run-time. ::

    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("-sin(5*x[0])", degree=2)


With :math:`u,v,f` and :math:`g`, we can write down the bilinear form
:math:`a` and the linear form :math:`L` (using UFL operators). ::

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

In order to transform our variational problem into a linear system we
need to assemble the coefficient matrix ``A`` and the right-side
vector ``b``. We do this using the function :py:meth:`assemble
<dolfin.cpp.fem.Assembler.assemble>`: ::

    # Assemble system
    A = assemble(a)
    b = assemble(L)

We specify a Vector for storing the result by defining a
:py:class:`Function <dolfin.cpp.function.Function>`. ::

    # Solution Function
    u = Function(V)

Next, we specify the iterative solver we want to use, in this case a
:py:class:`PETScKrylovSolver <dolfin.cpp.la.PETScKrylovSolver>` with
the conjugate gradient (CG) method, and attach the matrix operator to
the solver. ::

    # Create Krylov solver
    solver = PETScKrylovSolver("cg")
    solver.set_operator(A)

We impose our additional constraint by removing the null space
component from the solution vector. In order to do this we need a
basis for the null space. This is done by creating a vector that spans
the null space, and then defining a basis from it. The basis is then
attached to the matrix ``A`` as its null space. ::

    # Create vector that spans the null space and normalize
    null_vec = Vector(u.vector())
    V.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")

    # Create null space basis object and attach to PETSc matrix
    null_space = VectorSpaceBasis([null_vec])
    as_backend_type(A).set_nullspace(null_space)

Orthogonalization of ``b`` with respect to the null space makes sure
that it doesn't contain any component in the null space. ::

    null_space.orthogonalize(b);

Finally we are able to solve our linear system ::

    solver.solve(u.vector(), b)

and plot the solution ::

    plot(u, interactive=True)
