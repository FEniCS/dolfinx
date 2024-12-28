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
#
# This demo illustrates how to solve Poisson equation using a mixed
# (two-field) formulation and block-preconditioned iterative solver. In
# particular, it illustrates how to
#
# * Use mixed and non-continuous finite element spaces.
# * Set essential boundary conditions for subspaces and $H(\mathrm{div})$ spaces.
# * Construct a blocked linear system.
# * Use a block-preconditioned iterative linear solver.
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
# $\Sigma_h$ be the Brezzi-Douglas-Marini elements of polynomial order
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

import dolfinx.fem.petsc
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type, fem, io, la, mesh
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    div,
    exp,
    inner,
)

dtype = default_scalar_type
xdtype = default_real_type

# msh = mesh.create_unit_square(
#     MPI.COMM_WORLD, 32, 32, mesh.CellType.quadrilateral, dtype=default_real_type
# )
msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle, dtype=xdtype)

k = 1
V = fem.functionspace(msh, element("RT", msh.basix_cell(), k, dtype=xdtype))
W = fem.functionspace(msh, element("DG", msh.basix_cell(), k - 1, dtype=xdtype))

(sigma, u) = TrialFunction(V), TrialFunction(W)
(tau, v) = TestFunction(V), TestFunction(W)

x = SpatialCoordinate(msh)
f = 10.0 * exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)

dx = Measure("dx", msh)
a = [[inner(sigma, tau) * dx, inner(u, div(tau)) * dx], [inner(div(sigma), v) * dx, None]]
L = [
    inner(fem.Constant(msh, np.zeros(2, dtype=dtype)), tau) * dx,
    -inner(f, v) * dx,
]

a, L = fem.form(a, dtype=dtype), fem.form(L, dtype=dtype)


fdim = msh.topology.dim - 1
dofs_top = fem.locate_dofs_topological(
    V, fdim, mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
)
f_h1 = fem.Function(V, dtype=dtype)
f_h1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.sin(5 * x[0]))))
bc_top = fem.dirichletbc(f_h1, dofs_top)


dofs_bottom = fem.locate_dofs_topological(
    V, fdim, mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
)
f_h2 = fem.Function(V, dtype=dtype)
f_h2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), -np.sin(5 * x[0]))))
bc_bottom = fem.dirichletbc(f_h2, dofs_bottom)

bcs = [bc_top, bc_bottom]

# Assemble the matrix operator as a 'nested' matrix
A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
A.assemble()

# Define preconditioner
a_p00 = inner(sigma, tau) * dx + inner(div(sigma), div(tau)) * dx
a_p11 = inner(u, v) * dx
a_p = fem.form([[a_p00, None], [None, a_p11]])
P = fem.petsc.assemble_matrix_nest(a_p, bcs=bcs)
P.assemble()

# Assemble the RHS vector as a 'nested' vector and modify (apply
# lifting) to account for non-zero Dirichlet boundary conditions
b = fem.petsc.assemble_vector_nest(L)
fem.petsc.apply_lifting_nest(b, a, bcs=bcs)
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS vector
bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs)
fem.petsc.set_bc_nest(b, bcs0)

# Create PETSc Krylov solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A, P)
ksp.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
ksp.setType("minres")
ksp.setTolerances(rtol=1e-8)

# Set a field-split (block) preconditioner
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
nested_IS = P.getNestISs()
ksp.getPC().setFieldSplitIS(("sigma", nested_IS[0][0]), ("u", nested_IS[0][1]))
ksp_sigma, ksp_u = ksp.getPC().getFieldSplitSubKSP()
ksp_sigma.setType("preonly")
ksp_sigma.getPC().setType("lu")
if PETSc.Sys().hasExternalPackage("superlu_dist"):
    ksp_sigma.getPC().setFactorSolverType("superlu_dist")
ksp_u.setType("preonly")
ksp_u.getPC().setType("bjacobi")

# Solve
sigma, u = fem.Function(V, dtype=dtype), fem.Function(W, dtype=dtype)
x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(sigma.x), la.create_petsc_vector_wrap(u.x)])
ksp.solve(b, x)

with io.XDMFFile(msh.comm, "out_mixed_poisson/u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u)
