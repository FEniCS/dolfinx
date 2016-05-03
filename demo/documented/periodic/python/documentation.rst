.. Documentation for the Periodic Poisson demo from DOLFIN.

.. _demo_pde_periodic_python_documentation:


Poisson equation with periodic boundary conditions
==================================================

This demo is implemented in a single Python file,
:download:`demo_periodic.py`, which contains both the variational form
and the solver.

.. include:: ../common.txt


Implementation
--------------

This demo is implemented in a single file,
:download:`demo_periodic.py`.

First, the :py:mod:`dolfin` module is imported

.. code-block:: python

    from dolfin import *

A subclass of :py:class:`Expression <dolfin.cpp.function.Expression>`,
``Source``, is created for the source term ``f``. The function
:py:meth:`eval() <dolfin.cpp.fem.BasisFunction.eval>` returns values
for a function at the given point ``x``.

.. code-block:: python

    # Source term
    class Source(Expression):
        def eval(self, values, x):
            dx = x[0] - 0.5
            dy = x[1] - 0.5
            values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) \
                        + 1.0*exp(-(dx*dx + dy*dy)/0.02)

To define the boundaries, we create subclasses of the class
:py:class:`SubDomain <dolfin.cpp.mesh.SubDomain>`. A simple Python
function, returning a boolean, can be used to define the subdomain for
the Dirichlet boundary condition (:math:`\Gamma_D`). The function
should return True for those points inside the subdomain and False for
the points outside. In our case, we want to say that the points
:math:`(x, y)` such that :math:`y = 0` or :math:`y = 1` are inside of
:math:`\Gamma_D`. (Note that because of round-off errors, it is often
wise to instead specify :math:`y < \epsilon` or :math:`y > 1 -
\epsilon` where :math:`\epsilon` is a small number (such as machine
precision).)

.. code-block:: python

    # Sub domain for Dirichlet boundary condition
    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) \
                        and on_boundary)

The periodic boundary is defined by PeriodicBoundary and we define
what is inside the boundary in the same way as in
DirichletBoundary. The function ``map`` maps a coordinate ``x`` in
domain ``H`` to a coordinate ``y`` in the domain ``G``, it is used for
periodic boundary conditions, so that the right boundary of the domain
is mapped to the left boundary. When the class is defined, we create
the boundary by making an instance of the class.

.. code-block:: python

    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - 1.0
            y[1] = x[1]

A 2D mesh is created using the built-in class
:py:class:`UnitSquareMesh <dolfin.cpp.mesh.UnitSquareMesh>`, and we
define a finite element function space relative to this space. Notice
the fourth argument of :py:class:`FunctionSpace
<dolfin.cpp.function.FunctionSpace>`. It specifies that all functions
in ``V`` have periodic boundaries defined by ``pbc``. Also notice that
in order for periodic boundary conditions to work correctly it is necessary
that the mesh nodes on the periodic boundaries match up. This is automatically
satisfied for :py:class:`UnitSquareMesh <dolfin.cpp.mesh.UnitSquareMesh>`
but may require extra care with more general meshes (especially externally
generated ones).

.. code-block:: python

    # Create mesh and finite element
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

Now, we create the Dirichlet boundary condition using the class
:py:class:`DirichletBC <dolfin.cpp.fem.DirichletBC>`. A
:py:class:`DirichletBC <:py:class:`DirichletBC
<dolfin.cpp.fem.DirichletBC>` takes three arguments: the function
space the boundary condition applies to, the value of the boundary
condition, and the part of the boundary on which the condition
applies. In our example, the function space is ``V``, the value of the
boundary condition (0.0) can be represented using a
:py:class:`Constant <dolfin.functions.constant.Constant>` and the
Dirichlet boundary is defined by the class DirichletBoundary. The
definition of the Dirichlet boundary condition then looks as follows:

.. code-block:: python

    # Create Dirichlet boundary condition
    u0 = Constant(0.0)
    dbc = DirichletBoundary()
    bc0 = DirichletBC(V, u0, dbc)

When all boundary conditions are defined and created we can collect
them in a list:

.. code-block:: python

    # Collect boundary conditions
    bcs = [bc0]

Here only the Dirichlet boundary condition is put into the list
because the periodic boundary condition is already applied in the
definition of the function space. Next, we want to express the
variational problem. First, we need to specify the trial function u
and the test function v, both living in the function space V. We do
this by defining a :py:class:`TrialFunction
<dolfin.functions.function.TrialFunction>` and a
:py:class:`TestFunction <dolfin.functions.function.TestFunction>` on
the previously defined :py:class:`FunctionSpace
<dolfin.cpp.function.FunctionSpace>` V. The source function f is
created by making an instance of Source. With these ingredients, we
can write down the bilinear form a and the linear form L (using UFL
operators). In summary, this reads

.. code-block:: python

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Source(degree=1)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

Now, we have specified the variational form and can consider the solution of the variational problem. First, we need to define a :py:class:`Function <dolfin.cpp.function.Function>` u to represent the solution. (Upon initialization, it is simply set to the zero function.) A Function represents a function living in a finite element function space. Next, we can call the solve function with the arguments a == L, u and bcs as follows:

.. code-block:: python

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs)

The function u will be modified during the call to solve. The default settings for solving a variational problem have been used. However, the solution process can be controlled in much more detail if desired.

A :py:class:`Function <dolfin.cpp.function.Function>` can be manipulated in various ways, in particular, it can be plotted and saved to file. Here, we output the solution to a VTK file (using the suffix .pvd) for later visualization and also plot it using the plot command:

.. code-block:: python

    # Save solution to file
    file = File("periodic.pvd")
    file << u

    # Plot solution
    plot(u, interactive=True)

Complete code
-------------

.. literalinclude:: demo_periodic.py
    :start-after: # Begin demo
