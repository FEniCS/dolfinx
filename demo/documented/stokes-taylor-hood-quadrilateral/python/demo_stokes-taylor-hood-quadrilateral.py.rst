
.. _demo_pde_stokes-taylor-hood-quadrilateral_python_documentation:

Stokes equations with Taylor-Hood quadrilateral elements
==========================================

This demo is implemented in a single Python file,
:download:`demo_stokes-taylor-hood-quadrilateral.py`, which contains both the
variational form and the solver.

This demo shows the possibility to solve the Stokes equations on a quadrilateral mesh.

This demo illustrates how to:

* Create quadrilateral mesh
* Define function spaces for quadrilateral mesh
* Use mixed function spaces

Equation and problem definition
-------------------------------

Strong formulation
^^^^^^^^^^^^^^^^^^

.. math::
    - \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \ \Omega, \\
                    \nabla \cdot u &= 0 \quad {\rm in} \ \Omega. \\


.. note:: The sign of the pressure has been flipped from the classical
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

In this demo, we shall consider the lid-driven cavity problem on the unit square:

* :math:`\Omega = [0,1] \times [0,1]`
* :math:`u_0 = (1.0, 0.0)` for :math:`x_1 = 1` and :math:`u_0 = (0.0, 0.0)` otherwise
* :math:`f = (0.0, 0.0)`
* :math:`g = (0.0, 0.0)`

Implementation
--------------

First, the :py:mod:`dolfin` module is imported ::

    from dolfin import *

Then, quadrilateral mesh is generated with :py:class:`UnitQuadMesh
<dolfin.cpp.mesh.UnitQuadMesh>`
Mesh consisting of 32 x 32 squares on the unit square domain is created in the
following way ::

    # Create mesh
    mesh = UnitQuadMesh.create(32, 32)

Next, we define a :py:class:`FunctionSpace
<dolfin.functions.functionspace.FunctionSpace>` built on a mixed
finite element ``TH`` which consists of continuous
piecewise biquadratics (Q2) and continuous piecewise
bilinears (Q1). (This mixed finite element space is known as the Taylor-Hood
elements and is a stable, standard element pair for the Stokes
equations.)
``'Lagrange'``, ``'P'`` and ``'CG'`` finite element family tags are internally
translated to ``'Q'`` family if mesh consists of quadrilaterals or hexahedrons.
One may use ``'Q'`` tag directly instead of ``'Lagrange'``. ::

    # Define function spaces
    Q2 = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    Q1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    TH = Q2 * Q1
    W = FunctionSpace(mesh, TH)

Now that we have our mixed function space we
define boundary conditions ::

    # No-slip boundary condition for velocity
    # x0 = 0, x0 = 1, x1 = 0
    def no_slip_boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS
    noslip = Constant((0, 0))
    bc0 = DirichletBC(W.sub(0), noslip, no_slip_boundary)

    # Lid driven flow boundary condition for velocity
    # x1 = 1
    def lid_boundary(x):
        return x[1] > 1.0 - DOLFIN_EPS
    lid_flow = Constant((1, 0))
    bc1 = DirichletBC(W.sub(0), lid_flow, lid_boundary)

    # Collect boundary conditions
    bcs = [bc0, bc1]

The bilinear and linear forms corresponding to the weak mixed
formulation of the Stokes equations are defined as follows ::

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
deep copy. If no argument is given we will get a shallow copy. ::

    # Compute solution
    w = Function(W)
    solve(a == L, w, bcs)

    # Split the mixed solution using a shallow copy
    (u, p) = w.split()

Finally, we can store the solutions to files. ::

    # Save solution in VTK format
    ufile_pvd = File("velocity.pvd")
    ufile_pvd << u
    pfile_pvd = File("pressure.pvd")
    pfile_pvd << p

.. note:: The :py:func:`plot <dolfin.common.plot.plot>` command uses
          ``matplotlib`` backend by default, which does not support
          quadrilateral and hexahedral mesh.