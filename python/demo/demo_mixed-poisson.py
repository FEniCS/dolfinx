# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed formulation of the Poisson equation with block and
# monolithic preconditioners
#
# This demo illustrates how to solve the Poisson equation using a mixed
# (two-field) formulation and a block-preconditioned iterative solver.
# In particular, it illustrates how to
#
# * Use mixed and discontinuous finite element spaces.
# * Set essential boundary conditions for subspaces and
#   $H(\mathrm{div})$ spaces.
# * Construct a blocked linear system.
# * Construct a block-preconditioned iterative linear solver using
#   PETSc/petsc4y.
# * Construct a Hypre Auxiliary Maxwell Space (AMS) preconditioner for
#   $H(\mathrm{div})$ problems in two-dimensions.
# * Construct a monolithic Balancing Domain Decomposition by
#   Constraints (BDDC) preconditioner, following this
#   [paper](https://doi.org/10.1137/16M1080653).
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
# introducing an additional vector variable, the flux $\sigma = \nabla
# u$. The partial differential equations then read
#
# $$
# \begin{aligned}
#   \sigma - \nabla u &= 0 \quad {\rm in} \ \Omega, \\
#   \nabla \cdot \sigma &= - f \quad {\rm in} \ \Omega,
# \end{aligned}
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
# To solve the linear system for the mixed problem, we will use an
# iterative method with a block-diagonal preconditioner that is based on
# the Riesz map, see for example this
# [paper](https://doi.org/10.1002/(SICI)1099-1506(199601/02)3:1%3C1::AID-NLA67%3E3.0.CO;2-E).
#
# As an alternative, the demo also solves the same system
# monolithically with a BDDC preconditioner. BDDC is a non-overlapping
# domain decomposition method in which each MPI process owns one
# subdomain. It therefore requires the operator in *unassembled* form,
# i.e. assembled on each subdomain but not summed across processes.
# PETSc stores such operators as `MATIS`.

#
# ## Implementation
#
# Import the required modules:

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from basix.ufl import element
from dolfinx import fem, has_adios2, mesh
from dolfinx.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.mesh import CellType, GhostMode, create_unit_square

# Solution scalar (e.g., float32, complex128) and geometry (float32/64)
# types
dtype = PETSc.ScalarType
xdtype = PETSc.RealType
# -

# Create a two-dimensional mesh. The iterative solvers constructed later
# require special construction that is specific to two dimensions.
# Application in three-dimensions would require a number of changes to
# the linear solvers. The mesh is built without ghost cells, as BDDC
# needs each process to own a single, non-overlapping subdomain. The
# forms below have no interior facet integrals, so no ghosts are needed.

# +
msh = create_unit_square(
    MPI.COMM_WORLD, 96, 96, CellType.triangle, ghost_mode=GhostMode.none, dtype=xdtype
)
has_hypre = PETSc.Sys().hasExternalPackage("hypre")
hypre_ams_compatible = not np.issubdtype(dtype, np.complexfloating)
# -
#
# Here we construct compatible function spaces for the mixed Poisson
# problem. The `V` Raviart-Thomas ($\mathbb{RT}$) space is a
# vector-valued $H({\rm div})$ conforming space. The `W` space is a
# discontinuous Lagrange space of degree `k - 1`.
# ```{note}
# The $\mathbb{RT}_{k}$ element in DOLFINx/Basix is usually denoted as
# $\mathbb{RT}_{k-1}$ in the literature.
# ```
# The lowest-order case is $k=1$. The solvers below can be called with a
# higher degree, though convergence generally degrades as $k$ increases.
# For each degree, `build_forms` constructs the function spaces, the
# blocked variational form and the essential flux boundary conditions.
#
# The source is $f = 10\exp(-((x_0 - 0.5)^2 + (x_1 - 0.5)^2) / 0.02)$.
# The flux boundary condition $\sigma \cdot n = \sin(5x_0)$ is imposed on
# the top and bottom boundaries.


# +
def build_forms(msh: mesh.Mesh, k: int):
    """Build the function spaces, forms and boundary conditions.

    Args:
        msh: The mesh.
        k: Raviart-Thomas element degree.

    Returns:
        The flux space, the blocked bilinear form, the blocked linear
        form, the blocked Riesz-map preconditioner form, the boundary
        conditions and the solution functions.
    """
    if k < 1:
        raise ValueError("Element degree must be at least 1.")

    fdim = msh.topology.dim - 1
    facets_top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    facets_bottom = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    cells_top = mesh.compute_incident_entities(msh.topology, facets_top, fdim, fdim + 1)
    cells_bottom = mesh.compute_incident_entities(msh.topology, facets_bottom, fdim, fdim + 1)

    V = fem.functionspace(msh, element("RT", msh.basix_cell(), k, dtype=xdtype))
    W = fem.functionspace(msh, element("DG", msh.basix_cell(), k - 1, dtype=xdtype))
    Q = ufl.MixedFunctionSpace(V, W)
    sigma_trial, u_trial = ufl.TrialFunctions(Q)
    tau, v = ufl.TestFunctions(Q)

    x = ufl.SpatialCoordinate(msh)
    f = 10 * ufl.exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)
    dx = ufl.Measure("dx", msh)
    a: list[list[ufl.Form | None]] = ufl.extract_blocks(  # type: ignore[assignment]
        ufl.inner(sigma_trial, tau) * dx
        + ufl.inner(u_trial, ufl.div(tau)) * dx
        + ufl.inner(ufl.div(sigma_trial), v) * dx
    )
    L: list[ufl.Form] = [  # type: ignore[list-item]
        ufl.ZeroBaseForm((tau,)),
        -ufl.inner(f, v) * dx,
    ]
    a_p: list[list[ufl.Form | None]] = ufl.extract_blocks(  # type: ignore[assignment]
        ufl.inner(sigma_trial, tau) * dx
        + ufl.inner(ufl.div(sigma_trial), ufl.div(tau)) * dx
        + ufl.inner(u_trial, v) * dx
    )

    dofs_top = fem.locate_dofs_topological(V, fdim, facets_top)
    dofs_bottom = fem.locate_dofs_topological(V, fdim, facets_bottom)
    g = fem.Function(V, dtype=dtype)
    g.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.sin(5 * x[0]))), cells0=cells_top)
    g.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), -np.sin(5 * x[0]))), cells0=cells_bottom
    )
    bcs = [fem.dirichletbc(g, dofs_top), fem.dirichletbc(g, dofs_bottom)]

    sigma = fem.Function(V, name="sigma", dtype=dtype)
    u = fem.Function(W, name="u", dtype=dtype)
    return V, a, L, a_p, bcs, sigma, u


# -

# The block solver uses a `"nest"` matrix so that PETSc's
# [`fieldsplit`](https://petsc.org/release/manual/ksp/#sec-block-matrices)
# preconditioner can treat the $\sigma$ and $u$ fields separately. The
# $H({\rm div})$ block is preconditioned using either Hypre AMS or LU;
# the discontinuous Lagrange mass block uses Jacobi.
#
# Hypre AMS is available only for real scalar types. In two dimensions,
# it can precondition this $H({\rm div})$ problem because $H({\rm div})$
# and $H({\rm curl})$ are equivalent up to a rotation by $\pi/2$.


# +
def residual_monitor(label: str):
    """Build a KSP monitor that reports the residual on rank 0."""

    def monitor(_ksp, its: int, rnorm: float) -> None:
        if MPI.COMM_WORLD.rank == 0:
            print(f"{label}: iteration {its:>4d}, residual: {rnorm:.3e}")

    return monitor


def solve(k: int, use_hypre: bool) -> tuple[fem.Function, fem.Function]:
    """Solve the mixed Poisson problem with Raviart-Thomas degree ``k``.

    Args:
        k: Raviart-Thomas element degree.
        use_hypre: Whether to use Hypre AMS rather than LU.
    """
    if use_hypre and not has_hypre:
        raise RuntimeError("PETSc is not configured with Hypre.")
    if use_hypre and not hypre_ams_compatible:
        raise RuntimeError("Hypre AMS does not support complex scalar types.")

    V, a, L, a_p, bcs, sigma, u = build_forms(msh, k)
    problem = fem.petsc.LinearProblem(
        a,
        L,
        u=[sigma, u],
        P=a_p,
        kind="nest",
        bcs=bcs,
        petsc_options_prefix=f"demo_mixed_poisson_{k}_",
        petsc_options={
            "ksp_type": "minres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "ksp_rtol": 1e-5 if np.finfo(dtype).bits == 32 else 1e-7,
            "ksp_error_if_not_converged": True,
        },
    )
    ksp = problem.solver
    solver_label = f"k={k} ({'Hypre AMS' if use_hypre else 'LU'})"
    ksp.setMonitor(residual_monitor(solver_label))

    ksp_sigma, ksp_u = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.getPC().setType("jacobi")
    ksp_u.setFromOptions()
    pc_sigma = ksp_sigma.getPC()

    if use_hypre:
        pc_sigma.setType("hypre")
        pc_sigma.setHYPREType("ams")

        opts = PETSc.Options()
        opts[f"{ksp_sigma.prefix}pc_hypre_ams_cycle_type"] = 7  # type: ignore[index]
        opts[f"{ksp_sigma.prefix}pc_hypre_ams_relax_times"] = 3  # type: ignore[index]

        V_H1 = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), k, dtype=xdtype))
        V_curl = fem.functionspace(msh, element("N1curl", msh.basix_cell(), k, dtype=xdtype))
        G = discrete_gradient(V_H1, V_curl)
        G.assemble()
        pc_sigma.setHYPREDiscreteGradient(G)

        if k == 1:
            cvec0, cvec1 = fem.Function(V), fem.Function(V)
            cvec0.interpolate(lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[1]))))
            cvec1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[1]))))
            pc_sigma.setHYPRESetEdgeConstantVectors(cvec0.x.petsc_vec, cvec1.x.petsc_vec, None)
        else:
            V_H1d = fem.functionspace(msh, ("Lagrange", k, (msh.geometry.dim,)))
            Pi = interpolation_matrix(V_H1d, V)
            Pi.assemble()
            pc_sigma.setHYPRESetInterpolations(msh.geometry.dim, None, None, Pi, None)
            opts[f"{ksp_sigma.prefix}pc_hypre_ams_tol"] = 1e-12  # type: ignore[index]
            opts[f"{ksp_sigma.prefix}pc_hypre_ams_max_iter"] = 3  # type: ignore[index]

    else:
        pc_sigma.setType("lu")
        use_superlu = PETSc.IntType == np.int64
        if PETSc.Sys().hasExternalPackage("mumps") and not use_superlu:
            pc_sigma.setFactorSolverType("mumps")
        elif PETSc.Sys().hasExternalPackage("superlu_dist"):
            pc_sigma.setFactorSolverType("superlu_dist")

    ksp_sigma.setFromOptions()

    problem.solve()
    return sigma, u


# -

# The monolithic solver assembles the same system into a single matrix of
# kind `"is"` (PETSc `MATIS`), which stores the operator unassembled, and
# preconditions it with BDDC. The saddle-point structure needs a few
# non-default BDDC settings: the *benign trick* removes the pressure-like
# null space from the local subproblems, and the local Dirichlet, local
# Neumann and coarse solvers must handle symmetric indefinite matrices.
# Where PETSc is built without a suitable direct solver, the demo falls
# back to SVD, which is robust but only practical at this problem size.


# +
def solve_bddc(k: int) -> tuple[fem.Function, fem.Function]:
    """Solve the mixed Poisson problem monolithically with BDDC.

    Args:
        k: Raviart-Thomas element degree.
    """
    # Pick the sub-solvers from what PETSc has been configured with. SVD
    # is the fallback: it is robust but dense, and not a realistic choice
    # for large subdomains.
    local_solver, local_solver_type = "svd", "dummy"
    coarse_solver, coarse_solver_type = "svd", "dummy"
    if PETSc.Sys().hasExternalPackage("mumps"):
        local_solver, local_solver_type = "cholesky", "mumps"
        coarse_solver, coarse_solver_type = "cholesky", "mumps"
    elif PETSc.Sys().hasExternalPackage("superlu"):
        local_solver, local_solver_type = "lu", "superlu"
    elif PETSc.Sys().hasExternalPackage("umfpack"):
        local_solver, local_solver_type = "lu", "umfpack"
    if coarse_solver == "svd" and PETSc.Sys().hasExternalPackage("superlu_dist"):
        coarse_solver, coarse_solver_type = "lu", "superlu_dist"

    _, a, L, _, bcs, sigma, u = build_forms(msh, k)
    problem = fem.petsc.LinearProblem(
        a,
        L,
        u=[sigma, u],
        kind="is",
        bcs=bcs,
        petsc_options_prefix=f"demo_mixed_poisson_bddc_{k}_",
        petsc_options={
            "ksp_type": "gmres",
            "ksp_rtol": 1e-5 if np.finfo(dtype).bits == 32 else 1e-7,
            "ksp_error_if_not_converged": True,
            "pc_type": "bddc",
            "pc_bddc_use_local_mat_graph": False,
            "pc_bddc_benign_trick": None,
            "pc_bddc_nonetflux": None,
            "pc_bddc_detect_disconnected": None,
            "pc_bddc_dirichlet_pc_type": local_solver,
            "pc_bddc_dirichlet_pc_factor_mat_solver_type": local_solver_type,
            "pc_bddc_neumann_pc_type": local_solver,
            "pc_bddc_neumann_pc_factor_mat_solver_type": local_solver_type,
            "pc_bddc_coarse_pc_type": coarse_solver,
            "pc_bddc_coarse_pc_factor_mat_solver_type": coarse_solver_type,
        },
    )
    problem.solver.setMonitor(residual_monitor(f"k={k} (BDDC)"))
    problem.solve()
    return sigma, u


# -

# Solve and save the flux and scalar solutions for the lowest-order and
# next-order cases, using each solver in turn.

# +
if has_adios2:
    from dolfinx.io import VTXWriter
else:
    VTXWriter = None


def write(label: str, k: int, sigma: fem.Function, u: fem.Function) -> None:
    """Write the solution for degree ``k`` in VTX format."""
    if VTXWriter is None:
        return
    # VTX supports (discontinuous) Lagrange functions, so interpolate
    # the flux
    msh = sigma.function_space.mesh
    V_sigma = fem.functionspace(
        msh,
        element("DG", msh.basix_cell(), k, shape=(msh.geometry.dim,), dtype=xdtype),
    )
    sigma_output = fem.Function(V_sigma, name="sigma", dtype=dtype)
    sigma_output.interpolate(sigma)
    with VTXWriter(msh.comm, f"output_mixed_poisson_{label}_sigma_{k}.bp", sigma_output) as f:
        f.write(0.0)
    with VTXWriter(msh.comm, f"output_mixed_poisson_{label}_{k}.bp", u) as f:
        f.write(0.0)


use_hypre = has_hypre and hypre_ams_compatible
for k in (1, 2):
    write("block", k, *solve(k, use_hypre))
    write("bddc", k, *solve_bddc(k))

if not has_adios2:
    print("ADIOS2 required for VTX output.")
# -
