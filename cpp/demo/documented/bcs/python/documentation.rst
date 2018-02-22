.. Documentation for the bcs demo from DOLFIN.

.. _demo_pde_bcs_documentation:


Set boundary conditions for meshes that include boundary indicators
===================================================================

This demo is implemented in a single python file, :download:`demo_bcs.py`,
which contains both the variational form and the solver.


.. include:: ../common.txt

Implementation
--------------

This description goes through the implementation (in
:download:`demo_bcs.py`) of a solver for the above described Poisson
equation and how to set boundary conditions for a mesh that includes
boundary indicators.

First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

    from dolfin import *

Then, we import the mesh and create a finite element function space
:math:`V` relative to this mesh. In this case we create a
:py:class:`FunctionSpace<dolfin.functions.functionspace.FunctionSpace>`
``V`` consisting of continuous piecewise linear polynomials.

.. code-block:: python

    # Create mesh and define function space
    mesh = Mesh("../aneurysm.xml.gz")
    V = FunctionSpace(mesh, "CG", 1)

Now, we define the trial function u and the test function v, both living in the function space ``V``. We also define our variational problem, a and L. u and v are defined using the classes
:py:class:`TrialFunction <dolfin.functions.function.TrialFunction>` and
:py:class:`TestFunction<dolfin.functions.function.TrialFunction>`,
respetively, on the :py:class:`FunctionSpace<dolfin.functions.functionspace.FunctionSpace>` ``V``.
The source :math:`f` may be defined as a :py:class:`Constant <dolfin.functions.constant.Constant>`.
The bilinear and linear forms, ``a`` and ``L`` respectively, are defined using UFL operators.
Thus, the definition of the variational problem reads:

.. code-block:: python

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

Before we can solve the problem we must specify the boundary conditions.
We begin with specifying the values of the boundary conditions as :py:class:`Constant <dolfin.functions.constant.Constant>`s.
Then we use the class :py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>` to define the Dirichlet boundary conditions:

.. code-block:: python

    # Define boundary condition values
    u0 = Constant(0.0)
    u1 = Constant(1.0)
    u2 = Constant(2.0)
    u3 = Constant(3.0)

    # Define boundary conditions
    bc0 = DirichletBC(V, u0, 0)
    bc1 = DirichletBC(V, u1, 1)
    bc2 = DirichletBC(V, u2, 2)
    bc3 = DirichletBC(V, u3, 3)

:py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>` takes three arguments, the first one is our function space ``V``,
the next is the boundary condition value and the third is the subdomain indicator which is information stored in the mesh.

At this point we are ready to create a :py:class:`Function <dolfin.cpp.function.Function>` ``u`` to store the solution and call the solve function with the arguments
``a == L``, ``u``, ``[bc0, bc1, bc2, bc3]``, as follows:

.. code-block:: python

    # Compute solution
    u = Function(V)
    solve(a == L, u, [bc0, bc1, bc2, bc3])

When we have solved the problem, we save the solution to file and plot it.

.. code-block:: python

    # Write solution to file
    File("u.pvd") << u

    # Plot solution
    plot(u, interactive=True)


Complete code
-------------

.. literalinclude:: demo_bcs.py
   :start-after: # Begin demo

