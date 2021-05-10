# Cahn-Hilliard equation
# ======================
#
# This demo is implemented in a single Python file,
# :download:`demo_cahn-hilliard.py`, which contains both the variational
# forms and the solver.
#
# This example demonstrates the solution of the Cahn-Hilliard equation,
# a nonlinear time-dependent fourth-order PDE.
#
# * The built-in Newton solver
# * Use of the base class ``NonlinearProblem``
# * Automatic linearisation
# * A mixed finite element method
# * The :math:`\theta`-method for time-dependent equations
# * User-defined Expressions as Python classes
# * Form compiler options
# * Interpolation of functions
#
#
# Equation and problem definition
# -------------------------------
#
# The Cahn-Hilliard equation is a parabolic equation and is typically used
# to model phase separation in binary mixtures.  It involves first-order
# time derivatives, and second- and fourth-order spatial derivatives.  The
# equation reads:
#
# .. math::
#    \frac{\partial c}{\partial t} - \nabla \cdot M \left(\nabla\left(\frac{d f}{d c}
#              - \lambda \nabla^{2}c\right)\right) &= 0 \quad {\rm in} \ \Omega, \\
#    M\left(\nabla\left(\frac{d f}{d c} - \lambda \nabla^{2}c\right)\right) \cdot n
#    &= 0 \quad {\rm on} \ \partial\Omega, \\
#    M \lambda \nabla c \cdot n &= 0 \quad {\rm on} \ \partial\Omega.
#
# where :math:`c` is the unknown field, the function :math:`f` is usually
# non-convex in :math:`c` (a fourth-order polynomial is commonly used),
# :math:`n` is the outward directed boundary normal, and :math:`M` is a
# scalar parameter.
#
#
# Mixed form
# ^^^^^^^^^^
#
# The Cahn-Hilliard equation is a fourth-order equation, so casting it in
# a weak form would result in the presence of second-order spatial
# derivatives, and the problem could not be solved using a standard
# Lagrange finite element basis.  A solution is to rephrase the problem as
# two coupled second-order equations:
#
# .. math::
#    \frac{\partial c}{\partial t} - \nabla \cdot M \nabla\mu  &= 0 \quad {\rm in} \ \Omega, \\
#    \mu -  \frac{d f}{d c} + \lambda \nabla^{2}c &= 0 \quad {\rm in} \ \Omega.
#
# The unknown fields are now :math:`c` and :math:`\mu`. The weak
# (variational) form of the problem reads: find :math:`(c, \mu) \in V
# \times V` such that
#
# .. math::
#    \int_{\Omega} \frac{\partial c}{\partial t} q \, {\rm d} x
#    + \int_{\Omega} M \nabla\mu \cdot \nabla q \, {\rm d} x
#           &= 0 \quad \forall \ q \in V,  \\
#    \int_{\Omega} \mu v \, {\rm d} x - \int_{\Omega} \frac{d f}{d c} v \, {\rm d} x
#    - \int_{\Omega} \lambda \nabla c \cdot \nabla v \, {\rm d} x
#           &= 0 \quad \forall \ v \in V.
#
#
# Time discretisation
# ^^^^^^^^^^^^^^^^^^^
#
# Before being able to solve this problem, the time derivative must be
# dealt with. Apply the :math:`\theta`-method to the mixed weak form of
# the equation:
#
# .. math::
#
#    \int_{\Omega} \frac{c_{n+1} - c_{n}}{dt} q \, {\rm d} x
#    + \int_{\Omega} M \nabla \mu_{n+\theta} \cdot \nabla q \, {\rm d} x
#           &= 0 \quad \forall \ q \in V  \\
#    \int_{\Omega} \mu_{n+1} v  \, {\rm d} x - \int_{\Omega} \frac{d f_{n+1}}{d c} v  \, {\rm d} x
#    - \int_{\Omega} \lambda \nabla c_{n+1} \cdot \nabla v \, {\rm d} x
#           &= 0 \quad \forall \ v \in V
#
# where :math:`dt = t_{n+1} - t_{n}` and :math:`\mu_{n+\theta} =
# (1-\theta) \mu_{n} + \theta \mu_{n+1}`.  The task is: given
# :math:`c_{n}` and :math:`\mu_{n}`, solve the above equation to find
# :math:`c_{n+1}` and :math:`\mu_{n+1}`.
#
#
# Demo parameters
# ^^^^^^^^^^^^^^^
#
# The following domains, functions and time stepping parameters are used
# in this demo:
#
# * :math:`\Omega = (0, 1) \times (0, 1)` (unit square)
# * :math:`f = 100 c^{2} (1-c)^{2}`
# * :math:`\lambda = 1 \times 10^{-2}`
# * :math:`M = 1`
# * :math:`dt = 5 \times 10^{-6}`
# * :math:`\theta = 0.5`
#
#
# Implementation
# --------------
#
# This demo is implemented in the :download:`demo_cahn-hilliard.py`
# file.


import os

import numpy as np
from dolfinx import (Function, FunctionSpace, NewtonSolver, UnitSquareMesh,
                     log, plot)
from dolfinx.cpp.la import scatter_forward
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import NonlinearProblem
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (FiniteElement, TestFunctions, diff, dx, grad, inner, split,
                 variable)

try:
    import pyvista as pv
    import pyvistaqt as pvqt
    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)

except ModuleNotFoundError:
    print("pyvista is required to visualise the solution")
    have_pyvista = False

# Save all logging to file
log.set_output_file("log.txt")

#
# Next, various model parameters are defined::


# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06  # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# A unit square mesh with 97 (= 96 + 1) vertices in each direction is
# created, and on this mesh a
# :py:class:`FunctionSpace<dolfinx.fem.FunctionSpace>`
# ``ME`` is built using a pair of linear Lagrangian elements. ::

# Create mesh and build function space
mesh = UnitSquareMesh(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1 * P1)

# Trial and test functions of the space ``ME`` are now defined::

# Define test functions
q, v = TestFunctions(ME)

# .. index:: split functions
#
# For the test functions,
# :py:func:`TestFunctions<function ufl.argument.TestFunctions>` (note
# the 's' at the end) is used to define the scalar test functions ``q``
# and ``v``.
# Some mixed objects of the
# :py:class:`Function<dolfinx.fem.function.Function>` class on ``ME``
# are defined to represent :math:`u = (c_{n+1}, \mu_{n+1})` and :math:`u0
# = (c_{n}, \mu_{n})`, and these are then split into sub-functions::

# Define functions
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
c, mu = split(u)
c0, mu0 = split(u0)

# The line ``c, mu = split(u)`` permits direct access to the components of
# a mixed function. Note that ``c`` and ``mu`` are references for
# components of ``u``, and not copies.
#
# .. index::
#    single: interpolating functions; (in Cahn-Hilliard demo)
#
# The initial conditions are interpolated into a finite element space::

# Zero u
with u.vector.localForm() as x_local:
    x_local.set(0.0)

# Interpolate initial condition
u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - np.random.rand(x.shape[1])))

# The first line creates an object of type ``InitialConditions``.  The
# following two lines make ``u`` and ``u0`` interpolants of ``u_init``
# (since ``u`` and ``u0`` are finite element functions, they may not be
# able to represent a given function exactly, but the function can be
# approximated by interpolating it in a finite element space).
#
# .. index:: automatic differentiation
#
# The chemical potential :math:`df/dc` is computed using automated
# differentiation::

# Compute the chemical potential df/dc
c = variable(c)
f = 100 * c**2 * (1 - c)**2
dfdc = diff(f, c)

# The first line declares that ``c`` is a variable that some function can
# be differentiated with respect to. The next line is the function
# :math:`f` defined in the problem statement, and the third line performs
# the differentiation of ``f`` with respect to the variable ``c``.
#
# It is convenient to introduce an expression for :math:`\mu_{n+\theta}`::

# mu_(n+theta)
mu_mid = (1.0 - theta) * mu0 + theta * mu

# which is then used in the definition of the variational forms::

# Weak statement of the equations
F0 = inner(c, q) * dx - inner(c0, q) * dx + dt * inner(grad(mu_mid), grad(q)) * dx
F1 = inner(mu, v) * dx - inner(dfdc, v) * dx - lmbda * inner(grad(c), grad(v)) * dx
F = F0 + F1

# This is a statement of the time-discrete equations presented as part of
# the problem statement, using UFL syntax.

# .. index::
#    single: Newton solver; (in Cahn-Hilliard demo)
#
# The DOLFINX Newton solver requires a
# :py:class:`NonlinearProblem<dolfinx.fem.NonlinearProblem>` object to
# solve a system of nonlinear equations


# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6

# We can customize the linear solver used inside the NewtonSolver by modifying the
# PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# The setting of ``convergence_criterion`` to ``"incremental"`` specifies
# that the Newton solver should compute a norm of the solution increment
# to check for convergence (the other possibility is to use
# ``"residual"``, or to provide a user-defined check). The tolerance for
# convergence is specified by ``rtol``.
#
# To run the solver and save the output to a VTK file for later
# visualization, the solver is advanced in time from :math:`t_{n}` to
# :math:`t_{n+1}` until a terminal time :math:`T` is reached::

# Output file
file = XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w")
file.write_mesh(mesh)

# Step in time
t = 0.0

# Check if we are running on CI server and reduce run time
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 50 * dt

u.vector.copy(result=u0.vector)
scatter_forward(u.x)


# Prepare viewer for plotting solution during the computation
if have_pyvista:
    topology, cell_types = plot.create_vtk_topology(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    grid.point_arrays["u"] = u.sub(0).compute_point_values().real
    grid.set_active_scalars("u")
    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[0, 1])
    p.view_xy(True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

while (t < T):
    t += dt
    r = solver.solve(u)
    print(f"Step {int(t/dt)}: num iterations: {r[0]}")
    u.vector.copy(result=u0.vector)
    file.write_function(u.sub(0), t)

    # Update the plot window
    if have_pyvista:
        p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        grid.point_arrays["u"] = u.sub(0).compute_point_values().real
        p.app.processEvents()

file.close()

# Within the time stepping loop, the nonlinear problem is solved by
# calling :py:func:`solver.solve(problem,u.vector)<dolfinx.cpp.NewtonSolver.solve>`,
# with the new solution vector returned in :py:func:`u.vector<dolfinx.cpp.Function.vector>`.
# The solution vector associated with ``u`` is copied to ``u0`` at the
# end of each time step, and the ``c`` component of the solution
# (the first component of ``u``) is then written to file.

# Update ghost entries and plot
if have_pyvista:
    scatter_forward(u.x)
    grid.point_arrays["u"] = u.sub(0).compute_point_values().real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "u.png"
    pv.plot(grid, show_edges=True, screenshot=screenshot)
