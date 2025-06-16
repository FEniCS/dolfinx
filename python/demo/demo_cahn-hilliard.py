# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # Cahn-Hilliard equation
#
# This example demonstrates the solution of the Cahn-Hilliard equation,
# a nonlinear, time-dependent fourth-order PDE.
#
# - A mixed finite element method
# - The $\theta$-method for time-dependent equations
# - Automatic linearisation
# - Use of the class
#   {py:class}`NonlinearProblem<dolfinx.fem.petsc.NonlinearProblem>`
# - Interpolation of functions
# - Visualisation of a running simulation with
#   [PyVista](https://pyvista.org/)
#
# This demo is implemented in {download}`demo_cahn-hilliard.py`.
#
# ## Equation and problem definition
#
# The Cahn-Hilliard equation is a parabolic equation and is typically
# used to model phase separation in binary mixtures.  It involves
# first-order time derivatives, and second- and fourth-order spatial
# derivatives.  The equation reads:
#
# $$
# \begin{align}
# \frac{\partial c}{\partial t} -
#   \nabla \cdot M \left(\nabla\left(\frac{d f}{dc}
#   - \lambda \nabla^{2}c\right)\right) &= 0 \quad {\rm in} \ \Omega, \\
# M\left(\nabla\left(\frac{d f}{d c} -
#   \lambda \nabla^{2}c\right)\right) \cdot n
#   &= 0 \quad {\rm on} \ \partial\Omega, \\
# M \lambda \nabla c \cdot n &= 0 \quad {\rm on} \ \partial\Omega.
# \end{align}
# $$
#
# where $c$ is the unknown field, the function $f$ is usually non-convex
# in $c$ (a fourth-order polynomial is commonly used), $n$ is the
# outward directed boundary normal, and $M$ is a scalar parameter.
#
# ### Operator split form
#
# The Cahn-Hilliard equation is a fourth-order equation, so casting it
# in a weak form would result in the presence of second-order spatial
# derivatives, and the problem could not be solved using a standard
# Lagrange finite element basis.  A solution is to rephrase the problem
# as two coupled second-order equations:
#
# $$
# \begin{align}
# \frac{\partial c}{\partial t} - \nabla \cdot M \nabla\mu
#     &= 0 \quad {\rm in} \ \Omega, \\
# \mu -  \frac{d f}{d c} + \lambda \nabla^{2}c &= 0 \quad {\rm in}
#   \ \Omega.
# \end{align}
# $$
#
# The unknown fields are now $c$ and $\mu$. The weak (variational) form
# of the problem reads: find $(c, \mu) \in V \times V$ such that
#
# $$
# \begin{align}
# \int_{\Omega} \frac{\partial c}{\partial t} q \, {\rm d} x +
#     \int_{\Omega} M \nabla\mu \cdot \nabla q \, {\rm d} x
#     &= 0 \quad \forall \ q \in V,  \\
# \int_{\Omega} \mu v \, {\rm d} x - \int_{\Omega} \frac{d f}{d c} v \,
#   {\rm d} x - \int_{\Omega} \lambda \nabla c \cdot \nabla v \, {\rm d} x
#    &= 0 \quad \forall \ v \in V.
# \end{align}
# $$
#
# ### Time discretisation
#
# Before being able to solve this problem, the time derivative must be
# dealt with. Apply the $\theta$-method to the mixed weak form of the
# equation:
#
# $$
# \begin{align}
# \int_{\Omega} \frac{c_{n+1} - c_{n}}{dt} q \, {\rm d} x
#   + \int_{\Omega} M \nabla \mu_{n+\theta} \cdot \nabla q \, {\rm d} x
#   &= 0 \quad \forall \ q \in V  \\
# \int_{\Omega} \mu_{n+1} v  \, {\rm d} x - \int_{\Omega}
#   \frac{d f_{n+1}}{d c} v  \, {\rm d} x - \int_{\Omega} \lambda \nabla
#   c_{n+1} \cdot \nabla v \, {\rm d} x &= 0 \quad \forall \ v \in V
# \end{align}
# $$
#
# where $dt = t_{n+1} - t_{n}$ and $\mu_{n+\theta} = (1-\theta) \mu_{n} +
# \theta \mu_{n+1}$.
# The task is: given $c_{n}$ and $\mu_{n}$, solve the above equation to
# find $c_{n+1}$ and $\mu_{n+1}$.
#
# ### Demo parameters
#
# The following domains, functions and time stepping parameters are used
# in this demo:
#
# - $\Omega = (0, 1) \times (0, 1)$ (unit square)
# - $f = 100 c^{2} (1-c)^{2}$
# - $\lambda = 1 \times 10^{-2}$
# - $M = 1$
# - $dt = 5 \times 10^{-6}$
# - $\theta = 0.5$
#
# ## Implementation
#
# This demo is implemented in the {download}`demo_cahn-hilliard.py`
# file.

# +
import os

try:
    from petsc4py import PETSc

    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)

from mpi4py import MPI

import numpy as np

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square

try:
    import pyvista as pv
    import pyvistaqt as pvqt

    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

# Save all logging to file
log.set_output_file("log.txt")
# -

# Next, various model parameters are defined:

lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06  # time step
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson

# A unit square mesh with 96 cells edges in each direction is created,
# and on this mesh a
# {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>` `ME` is built
# using a pair of linear Lagrange elements.

msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
ME = functionspace(msh, mixed_element([P1, P1]))

# Trial and test functions of the space `ME` are now defined:

q, v = ufl.TestFunctions(ME)

# ```{index} split functions
# ```
#
# For the test functions, {py:func}`TestFunctions<function
# ufl.argument.TestFunctions>` (note the 's' at the end) is used to
# define the scalar test functions `q` and `v`. Some mixed objects of
# the {py:class}`Function<dolfinx.fem.function.Function>` class on `ME`
# are defined to represent $u = (c_{n+1}, \mu_{n+1})$ and $u0 = (c_{n},
# \mu_{n})$, and these are then split into sub-functions:

# +
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)
# -

# The line `c, mu = split(u)` permits direct access to the
# components of a mixed function. Note that `c` and `mu` are references
# for components of `u`, and not copies.
#
# ```{index} single: interpolating functions; (in Cahn-Hilliard demo)
# ```
#
# The initial conditions are interpolated into a finite element space:

# +
# Zero u
u.x.array[:] = 0.0

# Interpolate initial condition
rng = np.random.default_rng(42)
u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1])))
u.x.scatter_forward()
# -

# The first line creates an object of type `InitialConditions`.  The
# following two lines make `u` and `u0` interpolants of `u_init` (since
# `u` and `u0` are finite element functions, they may not be able to
# represent a given function exactly, but the function can be
# approximated by interpolating it in a finite element space).
#
# ```{index} automatic differentiation
# ```
#
# The chemical potential $df/dc$ is computed using UFL automatic
# differentiation:

# Compute the chemical potential df/dc
c = ufl.variable(c)
f = 100 * c**2 * (1 - c) ** 2
dfdc = ufl.diff(f, c)

# The first line declares that `c` is a variable that some function can
# be differentiated with respect to. The next line is the function $f$
# defined in the problem statement, and the third line performs the
# differentiation of `f` with respect to the variable `c`.
#
# It is convenient to introduce an expression for $\mu_{n+\theta}$:

# mu_(n+theta)
mu_mid = (1.0 - theta) * mu0 + theta * mu

# which is then used in the definition of the variational forms:

# Weak statement of the equations
F0 = (
    ufl.inner(c, q) * ufl.dx
    - ufl.inner(c0, q) * ufl.dx
    + dt * ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
)
F1 = (
    ufl.inner(mu, v) * ufl.dx
    - ufl.inner(dfdc, v) * ufl.dx
    - lmbda * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
)
F = F0 + F1

# This is a statement of the time-discrete equations presented as part
# of the problem statement, using UFL syntax.
#
# ```{index} single: Newton solver; (in Cahn-Hilliard demo)
# ```
#
# To solve the nonlinear system of equations,
# {py:class}`NonlinearProblem<dolfinx.fem.petsc.NonlinearProblem>` object
# to solve a system of nonlinear equations

# +
# For the factorisation of the underlying linearized problems, prefer
# MUMPS, then superlu_dist, then default.
# We measure convergence by looking at the norm of the increment of the
# solution between two iterations, called `stol` in PETSc, see:
# [`SNES convegence tests`](https://petsc.org/release/manual/snes/#convergence-tests)
# for further details.

use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
sys = PETSc.Sys()  # type: ignore
if sys.hasExternalPackage("mumps") and not use_superlu:
    linear_solver = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    linear_solver = "superlu_dist"
else:
    linear_solver = "petsc"
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
    "snes_atol": 0,
    "snes_rtol": 0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": linear_solver,
    "snes_monitor": None,
}
problem = NonlinearProblem(F, u, petsc_options=petsc_options)
# -

# To run the solver and save the output to a VTK file for later
# visualization, the solver is advanced in time from $t_{n}$ to
# $t_{n+1}$ until a terminal time $T$ is reached:

# +
# Output file
file = XDMFFile(MPI.COMM_WORLD, "demo_ch/output.xdmf", "w")
file.write_mesh(msh)

# Step in time
t = 0.0

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 50 * dt

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
V0, dofs = ME.sub(0).collapse()

# Prepare viewer for plotting the solution during the computation
if have_pyvista:
    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = u.x.array[dofs].real
    grid.set_active_scalars("c")

    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[0, 1])
    p.view_xy(True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

c = u.sub(0)
u0.x.array[:] = u.x.array
while t < T:
    t += dt
    _, converged_reason, num_iterations = problem.solve()
    print(f"Step {int(t / dt)}: {converged_reason=} {num_iterations=}")
    u0.x.array[:] = u.x.array
    file.write_function(c, t)

    # Update the plot window
    if have_pyvista:
        p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        grid.point_data["c"] = u.x.array[dofs].real
        p.app.processEvents()

file.close()

# Update ghost entries and plot
if have_pyvista:
    u.x.scatter_forward()
    grid.point_data["c"] = u.x.array[dofs].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "c.png"
    pv.plot(grid, show_edges=True, screenshot=screenshot)
