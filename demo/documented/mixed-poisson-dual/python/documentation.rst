.. Documentation for the dual mixed Poisson demo from DOLFIN.

.. _demo_pde_mixed-poisson-dual_python_documentation:

Dual-mixed formulation for Poisson equation
======================================

This demo is implemented in a single Python file,
:download:`demo_mixed-poisson-dual.py`, which contains both the
variational forms and the solver.

.. include:: ../common.txt


Implementation
--------------

This demo is implemented in the :download:`demo_mixed-poisson-dual.py`
file.

First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

    from dolfin import *

Then, we need to create a :py:class:`Mesh <dolfin.cpp.Mesh>` covering
the unit square. In this example, we will let the mesh consist of 32 x
32 squares with each square divided into two triangles:

.. code-block:: python

    # Create mesh
    mesh = UnitSquareMesh(32, 32)

.. index::
   pair: FunctionSpace; Discontinuous Raviart-Thomas
   pair: FunctionSpace; Lagrange

Next, we need to build the function space.

.. code-block:: python

    # Define finite elements spaces and build mixed space
    DRT = FiniteElement("DRT", mesh.ufl_cell(), 2)
    CG  = FiniteElement("CG", mesh.ufl_cell(), 3)
    W = FunctionSpace(mesh, DRT * CG)

The second argument to :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` specifies underlying
finite element, here mixed element obtained by ``*`` operator.

.. math::

    W = \{ (\tau, v) \ \text{such that} \ \tau \in DRT, v \in CG \}.

Next, we need to specify the trial functions (the unknowns) and the
test functions on this space. This can be done as follows

.. code-block:: python

    # Define trial and test functions
    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)

In order to define the variational form, it only remains to define the
source functions :math:`f` and :math:`g`. This is done just as for the
:ref:`mixed Poisson demo
<demo_pde_mixed-poisson_python_documentation>`:

.. code-block:: python

    # Define source functions
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("sin(5.0*x[0])", degree=2)

We are now ready to define the variational forms a and L.

.. code-block:: python

    # Define variational form
    a = (dot(sigma, tau) + dot(grad(u), tau) + dot(sigma, grad(v)))*dx
    L = - f*v*dx - g*v*ds

It only remains to prescribe the Dirichlet boundary condition for
:math:`u`. Essential boundary conditions are specified through the
class :py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>` which takes
three arguments: the function space the boundary condition is supposed
to be applied to, the data for the boundary condition, and the
relevant part of the boundary.

We want to apply the boundary condition to the second subspace of the
mixed space. Subspaces of a mixed :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` can be accessed
by the method :py:func:`sub
<dolfin.functions.functionspace.FunctionSpace.sub>`. In our case,
this reads ``W.sub(1)``. (Do *not* use the separate space ``CG`` as
this would mess up the numbering.)

Specifying the relevant part of the boundary can be done as for the
Poisson demo:

.. code-block:: python

    # Define Dirichlet BC
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

Now, all the pieces are in place for the construction of the essential
boundary condition:

.. code-block:: python

    bc = DirichletBC(W.sub(1), 0.0, boundary)

To compute the solution we use the bilinear and linear forms, and the
boundary condition, but we also need to create a :py:class:`Function
<dolfin.functions.function.Function>` to store the solution(s). The
(full) solution will be stored in the ``w``, which we initialise using
the :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` ``W``. The actual
computation is performed by calling :py:func:`solve
<dolfin.fem.solving.solve>`. The separate components ``sigma`` and
``u`` of the solution can be extracted by calling the :py:func:`split
<dolfin.functions.function.Function.split>` function. Finally, we plot
the solutions to examine the result.

.. code-block:: python

    # Compute solution
    w = Function(W)
    solve(a == L, w, bc)
    (sigma, u) = w.split()

    # Plot sigma and u
    plot(sigma)
    plot(u)
    interactive()


Complete code
-------------
.. literalinclude:: demo_mixed-poisson-dual.py
   :start-after: # Begin demo
