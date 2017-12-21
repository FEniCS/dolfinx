
.. _demo_pde_iterative_stokes_python_documentation:

Stokes equations with an iterative solver
=========================================

This demo is implemented in a single Python file,
:download:`demo_stokes-iterative.py`, which contains both the
variational forms and the solver.

This demo illustrates how to:

* Read mesh and subdomains from file
* Use mixed function spaces


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

In this demo, we shall consider the following definitions of the input
functions, the domain, and the boundaries:

* :math:`\Omega = [0,1]\times[0,1] \backslash {\rm dolphin}` (a unit
  cube)
* :math:`\Gamma_D =`
* :math:`\Gamma_N =`
* :math:`u_0 = (- \sin(\pi x_1), 0.0)` for :math:`x_0 = 1` and
  :math:`u_0 = (0.0, 0.0)` otherwise
* :math:`f = (0.0, 0.0)`
* :math:`g = (0.0, 0.0)`


Implementation
--------------

The Stokes equations as formulated above result in a system of linear
equations that is not positive-definite. Standard iterative linear
solvers typically fail to converge for such systems. Some care must
therefore be taken in preconditioning the systems of
equations. Moreover, not all of the linear algebra backends support
this. We therefore start by checking that either "PETSc" or "Tpetra"
(from Trilinos) is available. We also try to pick MINRES Krylov
subspace method which is suitable for symmetric indefinite problems.
If not available, costly QMR method is choosen. ::

    from dolfin import *

    # Test for PETSc or Tpetra
    if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
        info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
        exit()

    if not has_krylov_solver_preconditioner("amg"):
        info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
             "preconditioner, Hypre or ML.")
        exit()

    if has_krylov_solver_method("minres"):
        krylov_method = "minres"
    elif has_krylov_solver_method("tfqmr"):
        krylov_method = "tfqmr"
    else:
        info("Default linear algebra backend was not compiled with MINRES or TFQMR "
             "Krylov subspace method. Terminating.")
        exit()

Next, we define the mesh (a :py:class:`UnitCubeMesh
<dolfin.cpp.UnitCubeMesh>`) and a mixed finite element ``TH``.  Then
we build a :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` on this element.
(This mixed finite element space is known as the Taylor--Hood elements
and is a stable, standard element pair for the Stokes equations.) ::

    # Load mesh
    mesh = UnitCubeMesh.create(16, 16, 16, CellType.Type.hexahedron)

    # Build function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

Next, we define the boundary conditions. ::

    # Boundaries
    def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
    def left(x, on_boundary): return x[0] < DOLFIN_EPS
    def top_bottom(x, on_boundary):
        return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0, 0.0))
    bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

    # Inflow boundary condition for velocity
    inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=2)
    bc1 = DirichletBC(W.sub(0), inflow, right)

    # Collect boundary conditions
    bcs = [bc0, bc1]

The bilinear and linear forms corresponding to the weak mixed
formulation of the Stokes equations are defined as follows: ::

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0, 0.0))
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx


We can now use the same :py:class:`TrialFunctions
<dolfin.functions.function.TrialFunction>` and
:py:class:`TestFunctions <dolfin.functions.function.TestFunction>` to
define the preconditioner matrix. We first define the form
corresponding to the expression for the preconditioner (given in the
initial description above): ::

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v))*dx + p*q*dx

Next, we want to assemble the matrix corresponding to the bilinear
form and the vector corresponding to the linear form of the Stokes
equations. Moreover, we want to apply the specified boundary
conditions to the linear system. However, :py:func:`assembling
<dolfin.fem.assembling.assemble>` the matrix and vector and applying a
:py:func:`DirichletBC <dolfin.fem.bcs.DirichletBC>` separately will
possibly result in a non-symmetric system of equations. Instead, we
can use the :py:func:`assemble_system
<dolfin.fem.assembling.assemble_system>` function to assemble both the
matrix ``A``, the vector ``bb``, and apply the boundary conditions
``bcs`` in a symmetric fashion: ::

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

We do the same for the preconditioner matrix ``P`` using the linear
form ``L`` as a dummy form: ::

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs)

Next, we specify the iterative solver we want to use, in this case a
:py:class:`KrylovSolver <dolfin.cpp.KrylovSolver>`. We associate the
left-hand side matrix ``A`` and the preconditioner matrix ``P`` with
the solver by calling :py:func:`solver.set_operators
<dolfin.cpp.GenericLinearSolver.set_operators>`. ::

    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver(krylov_method, "amg")

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

We are now almost ready to solve the linear system of equations. It
remains to specify a :py:class:`Vector <dolfin.cpp.Vector>` for
storing the result. For easy manipulation later, we can define a
:py:class:`Function <dolfin.functions.function.Function>` and use the
vector associated with this Function. The call to
:py:func:`solver.solve <dolfin.cpp.KrylovSolver.solve>` then looks as
follows ::

    # Solve
    U = Function(W)
    solver.solve(U.vector(), bb)

Finally, we can play with the result in different ways: ::

    # Get sub-functions
    u, p = U.split()

    # Save solution in VTK format
    ufile_pvd = File("velocity.pvd")
    ufile_pvd << u
    pfile_pvd = File("pressure.pvd")
    pfile_pvd << p
