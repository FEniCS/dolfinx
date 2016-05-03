.. Documentation for the mixed Poisson demo from DOLFIN.

.. _demo_pde_mixed-poisson_python_documentation:

Mixed formulation for Poisson equation
======================================

This demo is implemented in a single Python file,
:download:`demo_mixed-poisson.py`, which contains both the variational
forms and the solver.

.. include:: ../common.txt


Implementation
--------------

This demo is implemented in the :download:`demo_mixed-poisson.py`
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
   pair: FunctionSpace; Brezzi-Douglas-Marini
   pair: FunctionSpace; Discontinous Lagrange

Next, we need to build the function space.

.. code-block:: python

    # Define finite elements spaces and build mixed space
    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, BDM * DG)

The second argument to :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` specifies underlying
finite element, here mixed element obtained by ``*`` operator.

.. math::

    W = \{ (\tau, v) \ \text{such that} \ \tau \in BDM, v \in DG \}.

Next, we need to specify the trial functions (the unknowns) and the
test functions on this space. This can be done as follows

.. code-block:: python

    # Define trial and test functions
    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)

In order to define the variational form, it only remains to define the
source function :math:`f`. This is done just as for the :ref:`Poisson
demo <demo_pde_poisson_python_documentation>`:

.. code-block:: python

    # Define source function
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)

We are now ready to define the variational forms a and L. Since,
:math:`u_0 = 0` in this example, the boundary term on the right-hand
side vanishes.

.. code-block:: python

    # Define variational form
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = - f*v*dx

It only remains to prescribe the boundary condition for the
flux. Essential boundary conditions are specified through the class
:py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>` which takes three
arguments: the function space the boundary condition is supposed to be
applied to, the data for the boundary condition, and the relevant part
of the boundary.

We want to apply the boundary condition to the first subspace of the
mixed space. Subspaces of a mixed :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` can be accessed
by the method :py:func:`sub
<dolfin.functions.functionspace.FunctionSpace.sub>`. In our case,
this reads ``W.sub(0)``. (Do *not* use the separate space ``BDM`` as
this would mess up the numbering.)

Next, we need to construct the data for the boundary condition. An
essential boundary condition is handled by replacing degrees of
freedom by the degrees of freedom evaluated at the given data. The
:math:`BDM` finite element spaces are vector-valued spaces and hence
the degrees of freedom act on vector-valued objects. The effect is
that the user is required to construct a :math:`G` such that :math:`G
\cdot n = g`.  Such a :math:`G` can be constructed by letting :math:`G
= g n`. In particular, it can be created by subclassing the
:py:class:`Expression <dolfin.functions.expression.Expression>`
class. Overloading the ``eval_cell`` method (instead of the usual
``eval``) allows us to extract more geometry information such as the
facet normals. Since this is a vector-valued expression, we also need
to overload the ``value_shape`` method.

.. index::
   single: Expression; (in Mixed Poisson demo)

.. code-block:: python

    # Define function G such that G \cdot n = g
    class BoundarySource(Expression):
           def __init__(self, mesh, **kwargs):
               self.mesh = mesh
           def eval_cell(self, values, x, ufc_cell):
               cell = Cell(self.mesh, ufc_cell.index)
               n = cell.normal(ufc_cell.local_facet)
               g = sin(5*x[0])
               values[0] = g*n[0]
               values[1] = g*n[1]
           def value_shape(self):
               return (2,)

     G = BoundarySource(mesh, degree=2)

Specifying the relevant part of the boundary can be done as for the
Poisson demo (but now the top and bottom of the unit square is the
essential boundary):

.. code-block:: python

    # Define essential boundary
    def boundary(x):
        return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

Now, all the pieces are in place for the construction of the essential
boundary condition:

.. code-block:: python

    bc = DirichletBC(W.sub(0), G, boundary)

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
.. literalinclude:: demo_mixed-poisson.py
   :start-after: # Begin demo
