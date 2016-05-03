.. Documentation for the auto adaptive poisson demo from DOLFIN.

.. _demo_pde_auto-adaptive-poisson_python_documentation:


Auto adaptive Poisson equation
==============================

This demo is implemented in a single Python file,
:download:`demo_auto-adaptive-poisson.py`, which contains both the
variational forms and the solver.

.. include:: ../common.txt

Implementation
--------------

This description goes through the implementation (in
:download:`demo_auto-adaptive-poisson.py`) of a solver for the above
described Poisson equation step-by-step.

First, the dolfin module is imported:

.. code-block:: python

    from dolfin import *

We begin by defining a mesh of the domain and a finite element
function space V relative to this mesh. We used the built-in mesh
provided by the class :py:class:`UnitSquareMesh
<dolfin.cpp.mesh.UnitSquareMesh>`. In order to create a mesh
consisting of 8 x 8 squares with each square divided into two
triangles, we do as follows:


.. code-block:: python

    # Create mesh and define function space
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)

The second argument to :py:class:`FunctionSpace
<dolfin.cpp.function.FunctionSpace>`, "Lagrange", is the finite
element family, while the third argument specifies the polynomial
degree. Thus, in this case, our space V consists of first-order,
continuous Lagrange finite element functions (or in order words,
continuous piecewise linear polynomials).

Next, we want to consider the Dirichlet boundary condition. In our
case, we want to say that the points (x, y) such that x = 0 or x = 1
are inside on the inside of :math:`\Gamma_D`. (Note that because of
rounding-off errors, it is often wise to instead specify :math:`x <
\epsilon` or :math:`x > 1 - \epsilon` where :math:`\epsilon` is a
small number (such as machine precision).)


.. code-block:: python

    # Define boundary condition
    u0 = Function(V)
    bc = DirichletBC(V, u0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")

Next, we want to express the variational problem. First, we need to
specify the trial function u and the test function v, both living in
the function space V. We do this by defining a
:py:class:`TrialFunction <dolfin.functions.function.TrialFunction>`
and a :py:class:`TestFunction
<dolfin.functions.function.TestFunction>` on the previously defined
:py:class:`FunctionSpace <dolfin.cpp.function.FunctionSpace>` V.

Further, the source f and the boundary normal derivative g are
involved in the variational forms, and hence we must specify
these. Both f and g are given by simple mathematical formulas, and can
be easily declared using the :py:class:`Expression
<dolfin.cpp.function.Expression>` class. Note that the strings
defining f and g use C++ syntax since, for efficiency, DOLFIN will
generate and compile C++ code for these expressions at run-time.

With these ingredients, we can write down the bilinear form a and the
linear form L (using UFL operators). In summary, this reads

.. code-block:: python

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",
                   degree=1)
    g = Expression("sin(5*x[0])", degree=1)
    a = inner(grad(u), grad(v))*dx()
    L = f*v*dx() + g*v*ds()

Now, we have specified the variational forms and can consider the
solution of the variational problem. First, we need to define a
:py:class:`Function <dolfin.cpp.function.Function>` u to represent the
solution. (Upon initialization, it is simply set to the zero
function.) A Function represents a function living in a finite element
function space.

.. code-block:: python

    # Define function for the solution
    u = Function(V)

Then define the goal functional:

.. code-block:: python

    # Define goal functional (quantity of interest)
    M = u*dx()

Next we specify the error tolerance for when the refinement shall stop:

.. code-block:: python

    # Define error tolerance
    tol = 1.e-5

Now, we have specified the variational forms and can consider the
solution of the variational problem. First, we define the
:py:class:`LinearVariationalProblem
<dolfin.cpp.fem.LinearVariationalProblem>` function with the arguments
a, L, u and bc. Next we send this problem to the
:py:class:`AdaptiveLinearVariationalSolver
<dolfin.cpp.fem.AdaptiveLinearVariationalSolver>` together with the
goal functional. Note that one may also choose several adaptations in
the error control. At last we solve the problem with the defined
tolerance:

.. code-block:: python

    # Solve equation a = L with respect to u and the given boundary
    # conditions, such that the estimated error (measured in M) is less
    # than tol
    problem = LinearVariationalProblem(a, L, u, bc)
    solver = AdaptiveLinearVariationalSolver(problem, M)
    solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
    solver.parameters["error_control"]["dual_variational_solver"]["symmetric"] = True
    solver.solve(tol)

    solver.summary()

    # Plot solution(s)
    plot(u.root_node(), title="Solution on initial mesh")
    plot(u.leaf_node(), title="Solution on final mesh")
    interactive()

Complete code
-------------

.. literalinclude:: demo_auto-adaptive-poisson.py
    :start-after: # Begin demo
