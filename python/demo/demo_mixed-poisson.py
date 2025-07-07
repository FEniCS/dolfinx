# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed formulation of the Poisson equation with a block-preconditioner/solver # noqa
#
# This demo illustrates how to solve the Poisson equation using a mixed
# (two-field) formulation and a block-preconditioned iterative solver.
# In particular, it illustrates how to
#
# * Use mixed and non-continuous finite element spaces.
# * Set essential boundary conditions for subspaces and
#   $H(\mathrm{div})$ spaces.
# * Construct a blocked linear system.
# * Construct a block-preconditioned iterative linear solver using
#   PETSc/petsc4y.
# * Construct a Hypre Auxiliary Maxwell Space (AMS) preconditioner for
#   $H(\mathrm{div})$ problems in two-dimensions.
#
# ```{admonition} Download sources
# :class: download
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
# where $n$ is the outward unit normal vector on the boundary. Looking
# at the variational form, we see that the boundary condition for the
# flux ($\sigma \cdot n = g$) is now an essential boundary condition
# (which should be enforced in the function space), while the other
# boundary condition ($u = u_0$) is a natural boundary condition (which
# should be applied to the variational form). Inserting the boundary
# conditions, this variational problem can be phrased in the general
# form: find $(\sigma, u) \in \Sigma_g \times V$ such that
#
# $$
#    a((\sigma, u), (\tau, v)) = L((\tau, v))
#    \quad \forall \ (\tau, v) \in \Sigma_0 \times V,
# $$
#
# where the variational forms $a$ and $L$ are defined as
#
# $$
#   a((\sigma, u), (\tau, v)) &:=
#     \int_{\Omega} \sigma \cdot \tau + \nabla \cdot \tau \ u
#   + \nabla \cdot \sigma \ v \ {\rm d} x, \\
#   L((\tau, v)) &:= - \int_{\Omega} f v \ {\rm d} x
#   + \int_{\Gamma_D} u_0 \tau \cdot n  \ {\rm d} s,
# $$
# and $\Sigma_g := \{ \tau \in H({\rm div})$ such that $\tau \cdot
# n|_{\Gamma_N} = g \}$ and $V := L^2(\Omega)$.
#
# To discretize the above formulation, two discrete function spaces
# $\Sigma_h \subset \Sigma$ and $V_h \subset V$ are needed to form a
# mixed function space $\Sigma_h \times V_h$. A stable choice of finite
# element spaces is to let $\Sigma_h$ be the Raviart-Thomas elements of
# polynomial order $k$ and let $V_h$ be discontinuous Lagrange elements of
# polynomial order $k-1$.
#
# To solve the linear system for the mixed problem, we will use am
# iterative method with a block-diagonal preconditioner that is based on
# the Riesz map, see for example this
# [paper](https://doi.org/10.1002/(SICI)1099-1506(199601/02)3:1%3C1::AID-NLA67%3E3.0.CO;2-E).

#
# ## Implementation
#
# Import the required modules:

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
import ufl
from basix.ufl import element
from dolfinx import fem, mesh
from dolfinx.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.mesh import CellType, create_unit_square

# Solution scalar (e.g., float32, complex128) and geometry (float32/64)
# types
dtype = dolfinx.default_scalar_type
xdtype = dolfinx.default_real_type
# -

# Create a two-dimensional mesh. The iterative solver constructed
# later requires special construction that is specific to two
# dimensions. Application in three-dimensions would require a number of
# changes to the linear solver.

# +
msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle, dtype=xdtype)
# -
#
# Here we construct compatible function spaces for the mixed Poisson
# problem. The `V` Raviart-Thomas ($\mathbb{RT}$) space is a
# vector-valued $H({\rm div})$ conforming space. The `W` space is a
# space of discontinuous Lagrange function of degree `k`.
# ```{note}
# The $\mathbb{RT}_{k}$ element in DOLFINx/Basix is usually denoted as
# $\mathbb{RT}_{k-1}$ in the literature.
# ```
# In the lowest-order case $k=1$. It can be increased, by the
# convergence of the iterative solver will degrade.

# +
k = 1
V = fem.functionspace(msh, element("RT", msh.basix_cell(), k, dtype=xdtype))
W = fem.functionspace(msh, element("Discontinuous Lagrange", msh.basix_cell(), k - 1, dtype=xdtype))
Q = ufl.MixedFunctionSpace(V, W)
# -

# Trial functions for $\sigma$ and $u$ are declared on the space $V$ and
# $W$, with corresponding test functions $\tau$ and $v$:

# +
sigma, u = ufl.TrialFunctions(Q)
tau, v = ufl.TestFunctions(Q)

# The source function is set to be $f = 10\exp(-((x_{0} - 0.5)^2 +
# (x_{1} - 0.5)^2) / 0.02)$:

# +
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)
# -

# We now declare the blocked bilinear and linear forms. We use
# `ufl.extract_blocks` to extract the block structure of the bilinear
# and linear form. For the first block of the right-hand side, we provide
# a form that efficiently is 0. We do this to preserve knowledge of the
# test space in the block. *Note that the defined `L` corresponds to
# $u_{0} = 0$ on $\Gamma_{D}$.*

# +
dx = ufl.Measure("dx", msh)

a = ufl.extract_blocks(
    ufl.inner(sigma, tau) * dx + ufl.inner(u, ufl.div(tau)) * dx + ufl.inner(ufl.div(sigma), v) * dx
)
L = [ufl.ZeroBaseForm((tau,)), -ufl.inner(f, v) * dx]

# -


# In preparation for Dirichlet boundary conditions, we use the function
# `locate_entities_boundary` to locate mesh entities (facets) with which
# degree-of-freedoms to be constrained are associated with, and then use
# `locate_dofs_topological` to get the  degree-of-freedom indices. Below
# we identify the degree-of-freedom in `V` on the (i) top ($x_{1} = 1$)
# and (ii) bottom ($x_{1} = 0$) of the mesh/domain.

# +
fdim = msh.topology.dim - 1
facets_top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
facets_bottom = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
dofs_top = fem.locate_dofs_topological(V, fdim, facets_top)
dofs_bottom = fem.locate_dofs_topological(V, fdim, facets_bottom)
# -

# Now, we create Dirichlet boundary objects for the condition $\sigma
# \cdot n = \sin(5 x_(0)$ on the top and bottom boundaries:

# +
cells_top_ = mesh.compute_incident_entities(msh.topology, facets_top, fdim, fdim + 1)
cells_bottom = mesh.compute_incident_entities(msh.topology, facets_bottom, fdim, fdim + 1)
g = fem.Function(V, dtype=dtype)
g.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.sin(5 * x[0]))), cells0=cells_top_)
g.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), -np.sin(5 * x[0]))), cells0=cells_bottom)
bcs = [fem.dirichletbc(g, dofs_top), fem.dirichletbc(g, dofs_bottom)]
# -

# Rather than solving the linear system $A x = b$, we will solve the
# preconditioned problem $P^{-1} A x = P^{-1} b$. Commonly $P = A$, but
# this does not lead to efficient solvers for saddle point problems.
#
# For this problem, we introduce the preconditioner
# $$
# a_p((\sigma, u), (\tau, v))
# = \begin{bmatrix} \int_{\Omega} \sigma \cdot \tau + (\nabla \cdot
# \sigma) (\nabla \cdot \tau) \ {\rm d} x  & 0 \\ 0 &
# \int_{\Omega} u \cdot v \ {\rm d} x \end{bmatrix}
# $$
# and assemble it into the matrix `P`:

# +
a_p = ufl.extract_blocks(
    ufl.inner(sigma, tau) * dx + ufl.inner(ufl.div(sigma), ufl.div(tau)) * dx + ufl.inner(u, v) * dx
)

# -

# We create finite element functions that will hold the $\sigma$ and $u$
# solutions:

# +
sigma, u = fem.Function(V, name="sigma", dtype=dtype), fem.Function(W, name="u", dtype=dtype)
# -

# We now create a linear problem solver for the mixed problem.
# As we will use different preconditions for the individual blocks of
# the saddle point problem, we specify the matrix kind to be "nest",
# so that we can use # [`fieldsplit`](https://petsc.org/release/manual/ksp/#sec-block-matrices)
# (block) type and set the 'splits' between the $\sigma$ and $u$ fields.


# +

problem = fem.petsc.LinearProblem(
    a,
    L,
    u=[sigma, u],
    P=a_p,
    kind="nest",
    bcs=bcs,
    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "ksp_rtol": 1e-8,
        "ksp_gmres_restart": 100,
        "ksp_view": "",
    },
)
# -


# +
ksp = problem.solver
ksp.setMonitor(
    lambda _, its, rnorm: PETSc.Sys.Print(f"Iteration: {its:>4d}, residual: {rnorm:.3e}")
)
# -

# For the $P_{11}$ block, which is the discontinuous Lagrange mass
# matrix, we let the preconditioner be the default, which is incomplete
# LU factorisation and which can solve the block exactly in one
# iteration. The $P_{00}$ requires careful handling as $H({\rm div})$
# problems require special preconditioners to be efficient.
#
# If PETSc has been configured with Hypre, we use the Hypre `Auxiliary
# Maxwell Space` (AMS) algebraic multigrid preconditioner. We can use
# AMS for this $H({\rm div})$-type problem in two-dimensions because
# $H({\rm div})$ and $H({\rm curl})$ spaces are effectively the same in
# two-dimensions, just rotated by $\pi/2.

# +
ksp_sigma, ksp_u = ksp.getPC().getFieldSplitSubKSP()
pc_sigma = ksp_sigma.getPC()
if PETSc.Sys().hasExternalPackage("hypre") and not np.issubdtype(dtype, np.complexfloating):
    pc_sigma.setType("hypre")
    pc_sigma.setHYPREType("ams")

    opts = PETSc.Options()
    opts[f"{ksp_sigma.prefix}pc_hypre_ams_cycle_type"] = 7
    opts[f"{ksp_sigma.prefix}pc_hypre_ams_relax_times"] = 2

    # Construct and set the 'discrete gradient' operator, which maps
    # grad H1 -> H(curl), i.e. the gradient of a scalar Lagrange space
    # to a H(curl) space
    V_H1 = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), k, dtype=xdtype))
    V_curl = fem.functionspace(msh, element("N1curl", msh.basix_cell(), k, dtype=xdtype))
    G = discrete_gradient(V_H1, V_curl)
    G.assemble()
    pc_sigma.setHYPREDiscreteGradient(G)

    assert k > 0, "Element degree must be at least 1."
    if k == 1:
        # For the lowest order base (k=1), we can supply interpolation
        # of the '1' vectors in the space V. Hypre can then construct
        # the required operators from G and the '1' vectors.
        cvec0, cvec1 = fem.Function(V), fem.Function(V)
        cvec0.interpolate(lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[1]))))
        cvec1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[1]))))
        pc_sigma.setHYPRESetEdgeConstantVectors(cvec0.x.petsc_vec, cvec1.x.petsc_vec, None)
    else:
        # For high-order spaces, we must provide the (H1)^d -> H(div)
        # interpolation operator/matrix
        V_H1d = fem.functionspace(msh, ("Lagrange", k, (msh.geometry.dim,)))
        Pi = interpolation_matrix(V_H1d, V)  # (H1)^d -> H(div)
        Pi.assemble()
        pc_sigma.setHYPRESetInterpolations(msh.geometry.dim, None, None, Pi, None)

        # High-order elements generally converge less well than the
        # lowest-order case with algebraic multigrid, so we perform
        # extra work at the multigrid stage
        opts[f"{ksp_sigma.prefix}pc_hypre_ams_tol"] = 1e-12
        opts[f"{ksp_sigma.prefix}pc_hypre_ams_max_iter"] = 3

    ksp_sigma.setFromOptions()
else:
    # If Hypre is not available, use LU factorisation on the $P_{00}$
    # block
    pc_sigma.setType("lu")
    use_superlu = PETSc.IntType == np.int64
    if PETSc.Sys().hasExternalPackage("mumps") and not use_superlu:
        pc_sigma.setFactorSolverType("mumps")
    elif PETSc.Sys().hasExternalPackage("superlu_dist"):
        pc_sigma.setFactorSolverType("superlu_dist")
# -

# Once we have set the preconditioners for the two blocks, we can
# solve the linear system. The `LinearProblem` class will
# automatically assemble the linear system, apply the boundary
# conditions, call the Krylov solver and update the solution
# vectors `u` and `sigma`.

# +
_1, converged_reason, _2 = problem.solve()
assert converged_reason > 0, f"Krylov solver has not converged, reason: {converged_reason}."
# -

# We save the solution `u` in VTX format:

# +
try:
    from dolfinx.io import VTXWriter

    u.name = "u"
    with VTXWriter(msh.comm, "output_mixed_poisson.bp", u, "bp4") as f:
        f.write(0.0)
except ImportError:
    print("ADIOS2 required for VTX output.")
# -
