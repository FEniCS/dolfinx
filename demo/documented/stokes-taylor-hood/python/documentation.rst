.. Documentation for the DOLFIN Stokes problem with Taylor-Hood elements demo

.. _demo_pde_stokes-taylor-hood_python_documentation:

Stokes equations with Taylor-Hood elements
==========================================

This demo is implemented in a single Python file,
:download:`demo_stokes-taylor-hood.py`, which contains both the
variational form and the solver.

.. include:: ../common.txt

Implementation
--------------

First, the :py:mod:`dolfin` module is imported:

.. code-block:: python

    from dolfin import *

In this example, different boundary conditions are prescribed on
different parts of the boundaries. This information must be made
available to the solver. One way of doing this, is to tag the
different sub-regions with different (integer) labels. DOLFIN provides
a class :py:class:`MeshFunction <dolfin.cpp.mesh.MeshFunction>` which
is useful for these types of operations: instances of this class
represent functions over mesh entities (such as over cells or over
facets). Mesh and mesh functions can be read from file in the
following way:

.. code-block:: python

    # Load mesh and subdomains
    mesh = Mesh("../dolfin_fine.xml.gz")
    sub_domains = MeshFunction("size_t", mesh, "../dolfin_fine_subdomains.xml.gz")

Next, we define a :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` built on a mixed
finite element ``TH`` which consists of continuous
piecewise quadratics and continuous piecewise
linears. (This mixed finite element space is known as the Taylor–Hood
elements and is a stable, standard element pair for the Stokes
equations.)

.. code-block:: python

    # Define function spaces
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

Now that we have our mixed function space and marked subdomains
defining the boundaries, we define boundary conditions:

.. code-block:: python

    # No-slip boundary condition for velocity
    # x1 = 0, x1 = 1 and around the dolphin
    noslip = Constant((0, 0))
    bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

    # Inflow boundary condition for velocity
    # x0 = 1
    inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
    bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

    # Collect boundary conditions
    bcs = [bc0, bc1]

Here, we have given four arguments in the call to
:py:class:`DirichletBC <dolfin.cpp.fem.DirichletBC>`. The first
specifies the :py:class:`FunctionSpace
<dolfin.cpp.function.FunctionSpace>`. Since we have a
mixed function space, we write
``W.sub(0)`` for the velocity component of the space, and
``W.sub(1)`` for the pressure componnen of the space.
The second argument specifies the value on the Dirichlet
boundary. The two last ones specifies the marking of the subdomains;
``sub_domains`` contains the subdomain markers and the number given as
the last argument is the subdomain index.

The bilinear and linear forms corresponding to the weak mixed
formulation of the Stokes equations are defined as follows:

.. code-block:: python

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0, 0))
    a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
    L = inner(f, v)*dx

To compute the solution we use the bilinear and linear forms, and the
boundary condition, but we also need to create a :py:class:`Function
<dolfin.cpp.function.Function>` to store the solution(s). The (full)
solution will be stored in w, which we initialize using the mixed
function space ``W``. The actual
computation is performed by calling solve with the arguments ``a``,
``L``, ``w`` and ``bcs``. The separate components ``u`` and ``p`` of
the solution can be extracted by calling the :py:meth:`split
<dolfin.functions.function.Function.split>` function. Here we use an
optional argument True in the split function to specify that we want a
deep copy. If no argument is given we will get a shallow copy. We want
a deep copy for further computations on the coefficient vectors.

.. code-block:: python

    # Compute solution
    w = Function(W)
    solve(a == L, w, bcs)

    # Split the mixed solution using deepcopy
    # (needed for further computation on coefficient vector)
    (u, p) = w.split(True)

We may be interested in the :math:`L^2` norms of u and p, they can be
calculated and printed by writing

.. code-block:: python

    print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

One can also split functions using shallow copies (which is enough
when we just plotting the result) by writing

.. code-block:: python

    # Split the mixed solution using a shallow copy
    (u, p) = w.split()

Finally, we can store to file and plot the solutions.

.. code-block:: python

    # Save solution in VTK format
    ufile_pvd = File("velocity.pvd")
    ufile_pvd << u
    pfile_pvd = File("pressure.pvd")
    pfile_pvd << p

    # Plot solution
    plot(u)
    plot(p)
    interactive()

Complete code
-------------

.. literalinclude:: demo_stokes-taylor-hood.py
    :start-after: # Begin demo
