# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed formulation for the Poisson equation

# This demo illustrates how to solve Poisson equation using a mixed
# (two-field) formulation. In particular, it illustrates how to
#
# * Use mixed and non-continuous finite element spaces.
# * Set essential boundary conditions for subspaces and $H(\mathrm{div})$ spaces.
#

# ```{admonition} Download sources
# :class: download
#
# * {download}`Python script <./demo_mixed-poisson.py>`
# * {download}`Jupyter notebook <./demo_mixed-poisson.ipynb>`
# ```
#
# ## Equation and problem definition
#
# An alternative formulation of Poisson equation can be formulated by
# introducing an additional (vector) variable, namely the (negative)
# flux: $\sigma = \nabla u$. The partial differential equations
# then read
#
# $$
# \begin{align}
#   \sigma - \nabla u &= 0 \quad {\rm in} \ \Omega, \\
#   \nabla \cdot \sigma &= - f \quad {\rm in} \ \Omega,
# \end{align}
# $$
# with boundary conditions
#
# $$
#   u = u_0 \quad {\rm on} \ \Gamma_{D},  \\
#   \sigma \cdot n = g \quad {\rm on} \ \Gamma_{N}.
# $$
#
# The same equations arise in connection with flow in porous media, and are
# also referred to as Darcy flow. Here $n$ denotes the outward pointing normal
# vector on the boundary. Looking at the variational form, we see that the
# boundary condition for the flux ($\sigma \cdot n = g$) is now an essential
# boundary condition (which should be enforced in the function space), while
# the other boundary condition ($u = u_0$) is a natural boundary condition
# (which should be applied to the variational form). Inserting the boundary
# conditions, this variational problem can be phrased in the general form: find
# $(\sigma, u) \in \Sigma_g \times V$ such that
#
# $$
#    a((\sigma, u), (\tau, v)) = L((\tau, v))
#    \quad \forall \ (\tau, v) \in \Sigma_0 \times V,
# $$
#
# where the variational forms $a$ and $L$ are defined as
#
# $$
# \begin{align}
#   a((\sigma, u), (\tau, v)) &=
#     \int_{\Omega} \sigma \cdot \tau + \nabla \cdot \tau \ u
#   + \nabla \cdot \sigma \ v \ {\rm d} x, \\
#   L((\tau, v)) &= - \int_{\Omega} f v \ {\rm d} x
#   + \int_{\Gamma_D} u_0 \tau \cdot n  \ {\rm d} s,
# \end{align}
# $$
# and $\Sigma_g = \{ \tau \in H({\rm div})$ such that $\tau \cdot n|_{\Gamma_N}
# = g \}$ and $V = L^2(\Omega)$.
#
# To discretize the above formulation, two discrete function spaces $\Sigma_h
# \subset \Sigma$ and $V_h \subset V$ are needed to form a mixed function space
# $\Sigma_h \times V_h$. A stable choice of finite element spaces is to let
# $\Sigma_h$ be the Brezzi-Douglas-Fortin-Marini elements of polynomial order
# $k$ and let $V_h$ be discontinuous elements of polynomial order $k-1$.
#
# We will use the same definitions of functions and boundaries as in the
# demo for {doc}`the Poisson equation <demo_poisson>`. These are:
#
# * $\Omega = [0,1] \times [0,1]$ (a unit square)
# * $\Gamma_{D} = \{(0, y) \cup (1, y) \in \partial \Omega\}$
# * $\Gamma_{N} = \{(x, 0) \cup (x, 1) \in \partial \Omega\}$
# * $u_0 = 0$
# * $g = \sin(5x)$   (flux)
# * $f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$  (source term)
#
# ## Implementation

# +

from mpi4py import MPI

import numpy as np

from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import Measure, SpatialCoordinate, TestFunctions, TrialFunctions, div, exp, inner

try:
    from petsc4py import PETSc

    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)


msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.quadrilateral)

k = 1
Q_el = element("BDMCF", msh.basix_cell(), k, dtype=default_real_type)
P_el = element("DG", msh.basix_cell(), k - 1, dtype=default_real_type)
V_el = mixed_element([Q_el, P_el])
V = fem.functionspace(msh, V_el)

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

x = SpatialCoordinate(msh)
f = 10.0 * exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)

dx = Measure("dx", msh)
a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
L = -inner(f, v) * dx

# Get subspace of V
V0 = V.sub(0)

fdim = msh.topology.dim - 1
facets_top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
Q, _ = V0.collapse()
dofs_top = fem.locate_dofs_topological((V0, Q), fdim, facets_top)


def f1(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = np.sin(5 * x[0])
    return values


f_h1 = fem.Function(Q)
f_h1.interpolate(f1)
bc_top = fem.dirichletbc(f_h1, dofs_top, V0)


facets_bottom = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
dofs_bottom = fem.locate_dofs_topological((V0, Q), fdim, facets_bottom)


def f2(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = -np.sin(5 * x[0])
    return values


f_h2 = fem.Function(Q)
f_h2.interpolate(f2)
bc_bottom = fem.dirichletbc(f_h2, dofs_bottom, V0)


bcs = [bc_top, bc_bottom]

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",
    },
)
try:
    w_h = problem.solve()
except PETSc.Error as e:  # type: ignore
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

sigma_h, u_h = w_h.split()

with io.XDMFFile(msh.comm, "out_mixed_poisson/u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_h)
