.. Documentation for the biharmonic demo from DOLFIN.

.. _demos_pde_biharmonic_python_documentation:

Biharmonic equation
===================

This demo is implemented in a single Python file, :download:`demo.py`,
which contains both the variational forms and the solver.

.. include:: ../common.txt


Implementation
--------------

This demo is implemented in the :download:`demo.py` file.

First, the ``dolfin`` module is imported:

.. code-block:: python

    from dolfin import *

Next, some parameters for the form compiler are set:

.. code-block:: python

    # Optimization options for the form compiler
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["optimize"] = True

A mesh is created, and a quadratic finite element function space:

.. code-block:: python

    # Create mesh and define function space
    mesh = UnitSquare(32, 32)
    V = FunctionSpace(mesh, "CG", 2)

A subclass of ``SubDomain``, ``DirichletBoundary`` is created for later
defining the boundary of the domian:

.. code-block:: python

    # Define Dirichlet boundary
    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

A subclass of ``Expression``, ``Source`` is created for the source
term :math:`f`:

.. code-block:: python

    class Source(Expression):
        def eval(self, values, x):
            values[0] = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])

The Dirichlet boundary condition is created:

.. code-block:: python

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, DirichletBoundary())

On the finite element space ``V``, trial and test functions are created:

.. code-block:: python

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

A function for the cell size :math:`h` is created, as is a function for
the average size of cells that share a facet (``h_avg``).  The UFL syntax
``('+')`` and ``('-')`` restricts a function to the ``('+')`` and ``('-')``
sides of a facet, respectively. The unit outward normal to cell boundaries (``n``)
is created, as is the source term ``f`` and the penalty parameter ``alpha``. The
penalty parameters is made a ``Constant`` so that it can be changed without needing
to regenerate code.

.. code-block:: python

    # Define normal component, mesh size and right-hand side
    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2.0
    n = FacetNormal(mesh)
    f = Source()

    # Penalty parameter
    alpha = Constant(8.0)

The bilinear and linear forms are defined:

.. code-block:: python

    # Define bilinear form
    a = inner(div(grad(u)), div(grad(v)))*dx \
      - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
      - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
      + alpha('+')/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS

    # Define linear form
    L = f*v*dx

A variational problem is created and solved:

.. code-block:: python

    # Create variational problem and solve
    problem = VariationalProblem(a, L, bc)
    u = problem.solve()

The computed solution is written to a file in VTK format and plotted to
the screen.

.. code-block:: python

    # Save solution to file
    file = File("biharmonic.pvd")
    file << u

    # Plot solution
    plot(u, interactive=True)


Complete code
-------------

.. literalinclude:: demo.py
   :start-after: # Begin demo
