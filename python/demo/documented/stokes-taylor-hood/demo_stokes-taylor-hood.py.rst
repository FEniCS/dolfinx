
.. _demo_pde_stokes-taylor-hood_python_documentation:

Stokes equations with Taylor-Hood elements
==========================================

This demo is implemented in a single Python file,
:download:`demo_stokes-taylor-hood.py`, which contains both the
variational form and the solver.

Equation and problem definition
-------------------------------

Strong formulation
^^^^^^^^^^^^^^^^^^

.. math::
	- \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \ \Omega, \\
                	\nabla \cdot u &= 0 \quad {\rm in} \ \Omega. \\


.. note::
        The sign of the pressure has been flipped from the classical
   	definition. This is done in order to have a symmetric (but not
	positive-definite) system of equations rather than a
	non-symmetric (but positive-definite) system of equations.

A typical set of boundary conditions on the boundary :math:`\partial
\Omega = \Gamma_{D} \cup \Gamma_{N}` can be:

.. math::
	u &= u_0 \quad {\rm on} \ \Gamma_{D}, \\
	\nabla u \cdot n + p n &= g \,   \quad\;\; {\rm on} \ \Gamma_{N}. \\


Weak formulation
^^^^^^^^^^^^^^^^

The Stokes equations can easily be formulated in a mixed variational
form; that is, a form where the two variables, the velocity and the
pressure, are approximated simultaneously. Using the abstract
framework, we have the problem: find :math:`(u, p) \in W` such that

.. math::
	a((u, p), (v, q)) = L((v, q))

for all :math:`(v, q) \in W`, where

.. math::

	a((u, p), (v, q))
				&= \int_{\Omega} \nabla u \cdot \nabla v
                 - \nabla \cdot v \ p
                 + \nabla \cdot u \ q \, {\rm d} x, \\
	L((v, q))
				&= \int_{\Omega} f \cdot v \, {\rm d} x
    			+ \int_{\partial \Omega_N} g \cdot v \, {\rm d} s. \\

The space :math:`W` should be a mixed (product) function space
:math:`W = V \times Q`, such that :math:`u \in V` and :math:`q \in Q`.

Domain and boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this demo, we shall consider the following definitions of the input functions, the domain, and the boundaries:

* :math:`\Omega = [0,1]\times[0,1] \backslash {\rm dolphin}` (a unit cube)
* :math:`\Gamma_D =`
* :math:`\Gamma_N =`
* :math:`u_0 = (- \sin(\pi x_1), 0.0)` for :math:`x_0 = 1` and :math:`u_0 = (0.0, 0.0)` otherwise
* :math:`f = (0.0, 0.0)`
* :math:`g = (0.0, 0.0)`


Implementation
--------------

In this example, different boundary conditions are prescribed on
different parts of the boundaries. Each sub-regions is tagged with
different (integer) labels. For this purpose, DOLFIN provides
a :py:class:`MeshFunction <dolfin.cpp.mesh.MeshFunction>` class
representing functions over mesh entities (such as over cells or over
facets). Mesh and mesh functions can be read from file in the
following way::

    import dolfin
    from dolfin import *

    # Load mesh and subdomains
    xdmf = XDMFFile(MPI.comm_world, "../dolfin_fine.xdmf")
    mesh = xdmf.read_mesh(MPI.comm_world)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    xdmf.read(sub_domains)

    cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry().ufc_coord_mapping = cmap


Next, we define a :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` built on a mixed
finite element ``TH`` which consists of continuous
piecewise quadratics and continuous piecewise
linears::

    # Define function spaces
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

The mixed finite element space is known as Taylor–Hood.
It is a stable, standard element pair for the Stokes
equations. Now we can define boundary conditions::

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

The first argument to
:py:class:`DirichletBC <dolfin.cpp.fem.DirichletBC>`
specifies the :py:class:`FunctionSpace
<dolfin.cpp.function.FunctionSpace>`. Since we have a
mixed function space, we write
``W.sub(0)`` for the velocity component of the space, and
``W.sub(1)`` for the pressure component of the space.
The second argument specifies the value on the Dirichlet
boundary. The last two arguments specify the marking of the subdomains:
``sub_domains`` contains the subdomain markers, and the final argument is the subdomain index.

The bilinear and linear forms corresponding to the weak mixed
formulation of the Stokes equations are defined as follows::

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0, 0))
    a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
    L = inner(f, v)*dx

We also need to create a :py:class:`Function
<dolfin.cpp.function.Function>` to store the solution(s). The (full)
solution will be stored in ``w``, which we initialize using the mixed
function space ``W``. The actual
computation is performed by calling solve with the arguments ``a``,
``L``, ``w`` and ``bcs``. The separate components ``u`` and ``p`` of
the solution can be extracted by calling the :py:meth:`split
<dolfin.functions.function.Function.split>` function. Here we use an
optional argument True in the split function to specify that we want a
deep copy. If no argument is given we will get a shallow copy. We want
a deep copy for further computations on the coefficient vectors::

    # Compute solution
    w = Function(W)
    solve(a == L, w, bcs)

    # Split the mixed solution using deepcopy
    # (needed for further computation on coefficient vector)
    (u, p) = w.split(True)

We can calculate the :math:`L^2` norms of u and p as follows::

    print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

    # Check pressure norm
    pnorm = p.vector().norm("l2")
    import numpy as np
    assert np.isclose(pnorm, 4116.91298427)

Finally, we can save and plot the solutions::

    # Save solution in XDMF format
    with XDMFFile(MPI.comm_world, "velocity.xdmf") as ufile_xdmf:
        ufile_xdmf.write(u)

    with XDMFFile(MPI.comm_world, "pressure.xdmf") as pfile_xdmf:
        pfile_xdmf.write(p)

    # Plot solution
    import matplotlib.pyplot as plt
    from dolfin.plotting import plot
    plt.figure()
    plot(u, title="velocity")

    plt.figure()
    plot(p, title="pressure")

    # Display plots
    plt.show()
