# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Domain decomposition solvers
#
# This demo illustrates how to:
#
# - Assemble an operator in PETSc's unassembled `MATIS` format.
# - Solve it with a Balancing Domain Decomposition by Constraints
#   (BDDC) preconditioner.
# - Set up a mesh so that each MPI process owns one non-overlapping
#   subdomain.
# - Attach the rigid body modes that BDDC needs to build an effective
#   coarse space for elasticity.
# - Solve both problems with PCHPDDM, an algebraic overlapping Schwarz
#   method whose coarse space is computed from local eigenproblems.
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_dd-solvers.py>`
# * {download}`Jupyter notebook <./demo_dd-solvers.ipynb>`
# ```
#
# ## Equations and problem definitions
#
# Two problems are solved on the same mesh of the unit square, each by
# its own function, so that the parts specific to a problem are
# separated from the domain decomposition machinery they share.
#
# The first is the Poisson equation with homogeneous Dirichlet
# conditions on the whole boundary,
#
# $$
# \begin{aligned}
#   - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
#   u &= 0 \quad {\rm on} \ \partial \Omega,
# \end{aligned}
# $$
#
# with $f = 2 \pi^{2} \sin(\pi x_{0}) \sin(\pi x_{1})$, for which the
# exact solution is $u = \sin(\pi x_{0}) \sin(\pi x_{1})$. The
# variational problem is: find $u \in V$ such that
#
# $$
# \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x
# = \int_{\Omega} f v \, {\rm d} x \quad \forall \ v \in V,
# $$
#
# with $V$ the space of piecewise linear Lagrange functions.
#
# The second is linearised elasticity for a displacement $u$, clamped
# on $x_{0} = 0$ and loaded by its own weight,
#
# $$
# \begin{aligned}
#   - \nabla \cdot \sigma(u) &= f \quad {\rm in} \ \Omega, \\
#   u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
#   \sigma(u) \cdot n &= 0 \quad {\rm on} \
#   \partial \Omega \setminus \Gamma_{D},
# \end{aligned}
# $$
#
# with $\sigma(u) = 2 \mu \, \epsilon(u) + \lambda \,
# {\rm tr}(\epsilon(u)) I$ and $\epsilon(u) = (\nabla u + (\nabla
# u)^{T}) / 2$. Its variational problem is: find $u \in W$ such that
#
# $$
# \int_{\Omega} \sigma(u) : \epsilon(v) \, {\rm d} x
# = \int_{\Omega} f \cdot v \, {\rm d} x \quad \forall \ v \in W,
# $$
#
# with $W$ the space of vector-valued piecewise linear Lagrange
# functions.
#
# ## Domain decomposition and the unassembled operator
#
# BDDC is a non-overlapping domain decomposition method. Each MPI
# process owns one subdomain, and the preconditioner is built from
# local solves on those subdomains plus a small coarse problem that
# couples them. It therefore needs the operator in *unassembled* form,
#
# $$
# A = \sum_{i} R_{i}^{T} A_{i} R_{i},
# $$
#
# where $A_{i}$ is the operator assembled on subdomain $i$ alone and
# $R_{i}$ restricts from the global to the local degrees of freedom.
# PETSc stores operators this way as `MATIS`, which DOLFINx creates
# when the matrix kind is `"is"`. Nothing is summed across processes:
# a matrix entry belonging to a degree of freedom on a subdomain
# interface is the sum of the contributions held by each process that
# shares it.
#
# Two consequences shape the demo.
#
# The mesh is created with `GhostMode.none`, so that the degrees of
# freedom a process holds are exactly those of the cells it owns. With
# ghost cells, a process would also hold degrees of freedom that
# receive no contribution from its own cells, leaving empty rows in
# $A_{i}$ and making the subdomain solves singular.
#
# Dirichlet degrees of freedom on a subdomain interface are shared. For
# $P_{1}$ elements the degrees of freedom sit at vertices, so wherever
# an interface meets the constrained boundary the vertex there is held
# by every process meeting at that point. Each of them must place a
# value on its local diagonal, or $A_{i}$ is again singular — but those
# values are summed, so each contributes a share rather than the whole
# value. The demo reports how many such degrees of freedom there are.
#
# ## Implementation

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import common, fem, mesh
from dolfinx.fem.petsc import (
    LinearProblem,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    set_bc,
)

dtype = PETSc.ScalarType
xdtype = PETSc.RealType
# -

# With CG, BDDC solves the subdomain interior problems once up front
# and assumes the interior residual then stays zero. In single
# precision rounding breaks this and CG breaks down, so both BDDC
# solves set `pc_bddc_switch_static` there, which redoes the interior
# correction in every application at the cost of an extra local solve.

# +
single_precision = np.finfo(dtype).bits == 32
rtol = 1e-5 if single_precision else 1e-8
PCOptions = dict[str, str | int | float | bool]
# -

# Both problems report how many of their constrained degrees of freedom
# lie on a subdomain interface, which
# :func:`~dolfinx.common.num_sharing_ranks` answers directly.


def norm_L2(v) -> float:
    """L2 norm of a UFL expression over the whole mesh."""
    form = fem.form(ufl.inner(v, v) * ufl.dx, dtype=dtype)
    return np.sqrt(form.mesh.comm.allreduce(fem.assemble_scalar(form), MPI.SUM).real)


def num_shared_dofs(V: fem.FunctionSpace, dofs: np.ndarray) -> int:
    """Count the entries of ``dofs`` held by more than one process.

    Args:
        V: Space the degrees of freedom belong to.
        dofs: Degrees of freedom to test, as local block indices.

    Returns:
        How many of ``dofs`` this process shares with another.
    """
    sharers = common.num_sharing_ranks(V.dofmap.index_map, dofs)
    return int(np.count_nonzero(sharers > 1))


# `solve_poisson` builds the Poisson problem on a given mesh and solves
# it with the preconditioner it is given, on a matrix of the kind that
# preconditioner needs: `MATIS` for BDDC, the default assembled matrix
# for PCHPDDM.


def solve_poisson(
    msh: mesh.Mesh, kind: str | None, pc_options: PCOptions
) -> tuple[fem.Function, int, int]:
    """Solve the Poisson problem on ``msh`` using CG.

    Args:
        msh: Mesh, which must have been built without ghost cells.
        kind: PETSc matrix kind, ``"is"`` for ``MATIS`` or ``None``
            for the default assembled matrix.
        pc_options: PETSc options that select and configure the
            preconditioner, without a prefix.

    Returns:
        The solution, the number of Krylov iterations, and the number
        of constrained degrees of freedom shared with another process.
    """
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 2 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    dofs = fem.locate_dofs_topological(V, tdim - 1, mesh.exterior_facet_indices(msh.topology))
    bcs = [fem.dirichletbc(dtype(0), dofs, V)]  # type: ignore[operator]

    uh = fem.Function(V, name="u", dtype=dtype)
    pc_type = pc_options["pc_type"]
    problem = LinearProblem(
        a,
        L,
        u=uh,
        bcs=bcs,
        kind=kind,
        petsc_options_prefix=f"demo_dd_poisson_{pc_type}_{V.dofmap.index_map.size_global}_",
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": rtol,
            "ksp_error_if_not_converged": True,
            **pc_options,
        },
    )
    problem.solve()
    return uh, problem.solver.getIterationNumber(), num_shared_dofs(V, dofs)


# Elasticity needs more than this. The rigid body modes — the
# displacements that cost no energy — span the kernel of the operator
# on a subdomain that the Dirichlet condition does not touch, and at
# any useful number of processes some subdomain is always interior.
# Poisson has the same difficulty in principle, but its kernel is one
# dimensional (the constants) and a single primal vertex removes it.
# In elasticity the kernel is three dimensional in 2D, and the vertices
# alone leave the rotation, so the subdomain solves stay singular.
#
# Two things fix it. PETSc reads the modes from the matrix as its
# *near* null space and uses them to build the coarse space. And
# `pc_bddc_use_change_of_basis` turns the edge constraints into
# explicit primal degrees of freedom, which are then removed from the
# subdomain problems; without it those problems keep their rigid body
# modes and the factorisation fails.

bddc_options: PCOptions = {
    "pc_type": "bddc",
    "pc_bddc_use_change_of_basis": True,
    "pc_bddc_switch_static": single_precision,
}

# PCHPDDM, PETSc's interface to the HPDDM library, takes a different
# route. It is an overlapping Schwarz method that works on the ordinary
# assembled matrix: the rows each process owns are extended into
# overlapping subdomains, and the preconditioner solves on every one of
# them. Its coarse space comes from local eigenproblems, in the spirit
# of GenEO: the eigenvectors with the smallest eigenvalues, which
# include the rigid body modes of a subdomain the Dirichlet condition
# does not touch, span the coarse space. No near null space or primal
# constraints are needed.
#
# The options select:
#
# - `pc_hpddm_harmonic_overlap`: build the eigenproblems algebraically,
#   on subdomains extended by one layer. It is required here: with only
#   an assembled matrix, PCHPDDM otherwise has no local operator to
#   build the eigenproblems from and falls back to one-level Schwarz.
# - `pc_hpddm_levels_1_eps_threshold_relative`: keep the eigenvectors
#   whose eigenvalues fall below a relative threshold, which is what
#   sizes the coarse space. How many to compute is left to PCHPDDM:
#   capping it with `pc_hpddm_levels_1_eps_nev` only costs iterations
#   here, and a cap too large for a subdomain makes the eigensolver
#   stall.
# - `pc_hpddm_levels_1_st_pc_type` and `pc_hpddm_levels_1_eps_pc_type`:
#   Cholesky factorisations inside the eigensolver.
# - `pc_hpddm_levels_1_pc_type` and `pc_hpddm_levels_1_pc_asm_overlap`:
#   additive Schwarz on subdomains overlapping by two layers, solved by
#   Cholesky factorisation (`pc_hpddm_levels_1_sub_pc_type`).
#   `pc_hpddm_define_subdomains` is off so that the Schwarz method
#   builds these subdomains itself, rather than reusing those of the
#   eigenproblems.
# - `pc_hpddm_levels_1_pc_asm_type` and `pc_hpddm_coarse_correction`:
#   symmetric variants of the Schwarz method and of the coarse
#   correction, which CG requires.
#
# PCHPDDM is available only when PETSc is configured with HPDDM and
# SLEPc.


hpddm_options: PCOptions = {
    "pc_type": "hpddm",
    "pc_hpddm_harmonic_overlap": 1,
    "pc_hpddm_levels_1_eps_threshold_relative": 100,
    "pc_hpddm_levels_1_st_pc_type": "cholesky",
    "pc_hpddm_levels_1_eps_pc_type": "cholesky",
    "pc_hpddm_define_subdomains": False,
    "pc_hpddm_levels_1_pc_type": "asm",
    "pc_hpddm_levels_1_pc_asm_overlap": 2,
    "pc_hpddm_levels_1_sub_pc_type": "cholesky",
    "pc_hpddm_levels_1_pc_asm_type": "basic",
    "pc_hpddm_coarse_correction": "balanced",
}


def rigid_body_modes(V: fem.FunctionSpace) -> PETSc.NullSpace:
    """Build the rigid body modes of a displacement space.

    Args:
        V: Vector-valued displacement space.

    Returns:
        The translations and rotations, which PETSc builds from the
        coordinates of the owned degrees of freedom.
    """
    gdim = V.mesh.geometry.dim
    num_owned = V.dofmap.index_map.size_local
    x = V.tabulate_dof_coordinates()[:num_owned, :gdim].copy()
    coords = PETSc.Vec().createWithArray(x.ravel(), bsize=gdim, comm=V.mesh.comm)  # type: ignore[arg-type]
    modes = PETSc.NullSpace().createRigidBody(coords)
    coords.destroy()
    return modes


# `solve_elasticity` assembles the operator and the right-hand side
# itself and drives a `PETSc.KSP` directly, rather than going through
# :class:`~dolfinx.fem.petsc.LinearProblem` as the Poisson solver does,
# to show what that class does on your behalf. It solves with CG and
# the preconditioner described by `pc_options`, on a matrix of the kind
# that preconditioner needs: `MATIS` for BDDC, the default assembled
# matrix for PCHPDDM.


def solve_elasticity(
    msh: mesh.Mesh, kind: str | None, pc_options: PCOptions
) -> tuple[fem.Function, int, int]:
    """Solve the elasticity problem on ``msh`` using CG.

    Args:
        msh: Mesh, which must have been built without ghost cells.
        kind: PETSc matrix kind, ``"is"`` for ``MATIS`` or ``None``
            for the default assembled matrix.
        pc_options: PETSc options that select and configure the
            preconditioner, without a prefix.

    Returns:
        The displacement, the number of Krylov iterations, and the
        number of constrained degrees of freedom shared with another
        process.
    """
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Non-dimensionalised, so that the matrix entries and the diagonal
    # written on constrained rows are of the same order
    ν = 0.3
    μ = 1.0 / (2.0 * (1.0 + ν))
    λ = ν / ((1.0 + ν) * (1.0 - 2.0 * ν))

    def σ(w):
        """Return an expression for the stress σ given a displacement w."""
        return 2.0 * μ * ufl.sym(ufl.grad(w)) + λ * ufl.div(w) * ufl.Identity(gdim)

    f = ufl.as_vector((0.0, -1.0e-2))
    a = fem.form(ufl.inner(σ(u), ufl.sym(ufl.grad(v))) * ufl.dx, dtype=dtype)
    L = fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)

    # Clamp the x0 = 0 edge
    tdim = msh.topology.dim
    facets = mesh.locate_entities_boundary(msh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    dofs = fem.locate_dofs_topological(V, tdim - 1, facets)
    bc = fem.dirichletbc(np.zeros(gdim, dtype=dtype), dofs, V)

    # Assemble the operator and the right-hand side
    A = assemble_matrix(a, bcs=[bc], kind=kind)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
    set_bc(b, [bc])

    # BDDC builds its coarse space from the rigid body modes. PCHPDDM
    # finds them itself, from the eigenproblems it solves on each
    # subdomain, so they are neither built nor attached for it.
    pc_type = pc_options["pc_type"]
    if pc_type == "bddc":
        # PETSc holds a reference once the modes are attached, so this
        # side keeps none
        modes = rigid_body_modes(V)
        A.setNearNullSpace(modes)
        modes.destroy()
    A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore[arg-type]

    # Everything, Krylov method included, goes through the options
    # database under a prefix of this solver's own, so that nothing is
    # left behind for the next solve to inherit
    options: PCOptions = {
        "ksp_type": "cg",
        "ksp_rtol": rtol,
        "ksp_max_it": 100,
        "ksp_error_if_not_converged": True,
        **pc_options,
    }
    prefix = f"demo_dd_elasticity_{pc_type}_{V.dofmap.index_map.size_global}_"
    ksp = PETSc.KSP().create(msh.comm)  # type: ignore[arg-type]
    ksp.setOperators(A)
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options(prefix)
    for key, value in options.items():
        opts[key] = value  # type: ignore[index]
    ksp.setFromOptions()

    uh = fem.Function(V, name="u", dtype=dtype)
    ksp.solve(b, uh.x.petsc_vec)

    # Cleared after the solve, not before: PCHPDDM reads some of its
    # options when it is first applied rather than at setFromOptions
    for key in options:
        del opts[key]  # type: ignore[arg-type]
    uh.x.scatter_forward()

    its = ksp.getIterationNumber()
    num_shared = num_shared_dofs(V, dofs)
    ksp.destroy()
    A.destroy()
    b.destroy()
    return uh, its, num_shared


# The number of iterations of both methods is close to independent of
# the mesh size, so refining the mesh does not slow convergence the way
# it would for a one-level method. Solving on a sequence of meshes
# shows this. Each mesh is built once and handed to every solver. Both
# problems are also solved with PCHPDDM when PETSc provides it and
# there is more than one process; with one there is no decomposition.


# +
def report(label, n, uh, its, shared, pc_options, metric, value) -> None:
    """Print one solver's result, on rank 0 only."""
    V = uh.function_space
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    shared = V.mesh.comm.allreduce(shared, MPI.SUM)
    if V.mesh.comm.rank == 0:
        pc_name = str(pc_options["pc_type"]).upper()
        print(
            f"{label}, n = {n:>3d}: {num_dofs:>7d} dofs, {its:>3d} CG iterations "
            f"({pc_name}), {metric} = {value:.3e}, shared Dirichlet dofs = {shared}"
        )


comm = MPI.COMM_WORLD
preconditioners: list[tuple[str | None, PCOptions]] = [("is", bddc_options)]
if PETSc.Sys.hasExternalPackage("hpddm") and comm.size > 1:
    preconditioners.append((None, hpddm_options))

for n in (32, 64):
    # BDDC requires one non-overlapping subdomain per process, so the
    # mesh is built without ghost cells
    msh = mesh.create_unit_square(
        comm, n, n, mesh.CellType.triangle, ghost_mode=mesh.GhostMode.none, dtype=xdtype
    )

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    for kind, pc_options in preconditioners:
        uh, its, shared = solve_poisson(msh, kind, pc_options)
        report("Poisson   ", n, uh, its, shared, pc_options, "L2 error", norm_L2(uh - u_exact))

    for kind, pc_options in preconditioners:
        uh, its, shared = solve_elasticity(msh, kind, pc_options)
        report("Elasticity", n, uh, its, shared, pc_options, "|u|_L2", norm_L2(uh))
# -

# With more than one process, `shared Dirichlet dofs` is non-zero: those
# are the constrained vertices where a subdomain interface meets the
# Dirichlet boundary, and they are the reason the boundary condition
# value has to be distributed across the processes that share them
# rather than written by the owner alone. The Poisson problem is
# constrained on the whole boundary and the elasticity problem on one
# edge, so the counts differ.
