# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Geometric multigrid for the Poisson equation
#
# This demo illustrates how to:
#
# - Build a hierarchy of uniformly refined meshes
# - Create the interpolation matrix between spaces on different meshes
#   with {py:func}`nonmatching_interpolation_matrix
#   <dolfinx.fem.petsc.nonmatching_interpolation_matrix>`
# - Use these matrices as prolongations in PETSc's geometric multigrid
#   preconditioner ({py:attr}`PETSc.PC.Type.MG
#   <petsc4py.PETSc.PC.Type.MG>`)
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_poisson-gmg.py>`
# * {download}`Jupyter notebook <./demo_poisson-gmg.ipynb>`
# ```
#
# ## Problem definition
#
# On the unit square $\Omega = (0, 1)^2$ we solve
#
# $$
# \begin{aligned}
#   -\nabla^2 u &= f && \text{in } \Omega, \\
#   u &= u_{\mathrm{ex}} && \text{on } \partial\Omega,
# \end{aligned}
# $$
#
# where $f = -\nabla^2 u_{\mathrm{ex}}$ for the manufactured solution
#
# $$
# u_{\mathrm{ex}}(x, y) = \sin(\pi x)\sin(2\pi y) + xy.
# $$
#
# The variational problem reads: find $u \in V$ with $u = u_{\mathrm{ex}}$
# on $\partial\Omega$ such that
#
# $$
# \int_\Omega \nabla u \cdot \nabla v ~\mathrm{d}x
# = \int_\Omega f v ~\mathrm{d}x \quad \forall v \in V_0,
# $$
#
# where $V_0 \subset V$ is the subspace of functions that vanish on
# $\partial\Omega$.
#
# ## Geometric multigrid
#
# Geometric multigrid solves a problem on a sequence of meshes. On each
# mesh a few iterations of a cheap smoother reduce the oscillatory part
# of the error. The remaining smooth part is restricted to the next
# coarser mesh, where it can be represented, and corrected there. The
# meshes are connected by a *prolongation* $P_\ell$ from level $\ell$ to
# level $\ell + 1$. With a suitable smoother, the number of Krylov
# iterations is independent of the mesh size.
#
# The operator $A_\ell$ on a coarse level can be obtained in two ways:
#
# - by a *Galerkin projection* of the operator on the next finer level,
#   $A_\ell = P_\ell^T A_{\ell + 1} P_\ell$, computed algebraically from
#   the fine matrix, or
# - by *rediscretisation*: assembling the bilinear form on the coarse
#   mesh, with the boundary conditions of that mesh.
#
# This demo shows both.
#
# For nested Lagrange spaces on a mesh and its refinement, the
# prolongation is the interpolation of the coarse basis functions into
# the fine space. As the two spaces are defined on different meshes,
# it is computed with
# {py:func}`nonmatching_interpolation_matrix
# <dolfinx.fem.petsc.nonmatching_interpolation_matrix>`, which evaluates
# the coarse basis functions at the interpolation points of the fine
# space, also when the meshes are distributed differently in parallel.
#
# ## Implementation
#
# We start by importing the required modules.

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx.fem.petsc
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, la, mesh

# -

# The hierarchy is a coarse mesh refined uniformly a number of times,
# with a continuous piecewise linear space on each mesh.


def create_hierarchy(num_levels: int) -> list[fem.FunctionSpace]:
    """Function spaces on a coarse mesh and its uniform refinements."""
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dtype=default_real_type)
    spaces = [fem.functionspace(msh, ("Lagrange", 1))]
    for _ in range(num_levels - 1):
        msh.topology.create_entities(1)
        msh, _, _ = mesh.refine(msh)
        spaces.append(fem.functionspace(msh, ("Lagrange", 1)))
    return spaces


# The multigrid preconditioner builds its hierarchy by repeatedly
# coarsening the {py:class}`PETSc.DM <petsc4py.PETSc.DM>` attached to it
# with {py:meth}`PETSc.PC.setDM <petsc4py.PETSc.PC.setDM>`
# ({py:meth}`PETSc.DM.coarsen <petsc4py.PETSc.DM.coarsen>`), and asks each
# coarse `DM` for the interpolation to the next finer one
# ({py:meth}`PETSc.DM.createInterpolation
# <petsc4py.PETSc.DM.createInterpolation>`). We represent each level by a
# {py:class}`PETSc.DMShell <petsc4py.PETSc.DMShell>` with the parallel
# layout of its function space, set with
# {py:meth}`setGlobalVector <petsc4py.PETSc.DMShell.setGlobalVector>`. Its
# coarsening, set with
# {py:meth}`setCoarsen <petsc4py.PETSc.DMShell.setCoarsen>`, returns the
# `DMShell` of the next coarser level, and its interpolation, set with
# {py:meth}`setCreateInterpolation
# <petsc4py.PETSc.DMShell.setCreateInterpolation>`, is the prolongation
# computed with {py:func}`nonmatching_interpolation_matrix
# <dolfinx.fem.petsc.nonmatching_interpolation_matrix>`.


# +
def shell_dm(V: fem.FunctionSpace) -> PETSc.DMShell:
    """DMShell with the parallel layout of the space V."""
    dm = PETSc.DMShell().create(V.mesh.comm)  # type: ignore[arg-type]
    dm.setGlobalVector(la.petsc.create_vector([(V.dofmap.index_map, V.dofmap.index_map_bs)]))
    return dm


def attach_hierarchy(
    pc: PETSc.PC, spaces: list[fem.FunctionSpace]
) -> tuple[list[PETSc.DMShell], list[PETSc.Mat]]:
    """Give pc a DM that PCMG can coarsen through the spaces.

    Returns the DMs and prolongations, which must outlive the solve.
    """
    dms = [shell_dm(V) for V in spaces]
    prolongations = []
    for coarse, fine, dm_c, dm_f in zip(spaces[:-1], spaces[1:], dms[:-1], dms[1:], strict=True):
        P = dolfinx.fem.petsc.nonmatching_interpolation_matrix(coarse, fine)
        P.assemble()
        prolongations.append(P)
        dm_f.setCoarsen(lambda dm, comm, dm_c=dm_c: dm_c)
        dm_c.setCreateInterpolation(lambda dmc, dmf, P=P: (P, None))  # type: ignore[arg-type]
    pc.setDM(dms[-1])
    return dms, prolongations


# -

# ### Rediscretised coarse operators
#
# For rediscretisation, each level assembles the bilinear form on its
# own mesh. The coarse levels solve for a *correction*, which vanishes on
# the Dirichlet boundary, so their boundary condition is homogeneous.


def level_problem(V: fem.FunctionSpace) -> tuple[fem.Form, fem.DirichletBC]:
    """Bilinear form and homogeneous Dirichlet condition on the space V."""
    msh = V.mesh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    dofs = fem.locate_dofs_topological(V, tdim - 1, mesh.exterior_facet_indices(msh.topology))
    bc = fem.dirichletbc(default_scalar_type(0), dofs, V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=default_scalar_type)
    return a, bc


# Without the Galerkin projection, the multigrid preconditioner asks the
# `DM` of each coarse level for its operator: it creates the matrix with
# the `DM`'s matrix-creation callback
# ({py:meth}`setCreateMatrix <petsc4py.PETSc.DMShell.setCreateMatrix>`)
# and fills it with its operator callback
# ({py:meth}`setKSPComputeOperators
# <petsc4py.PETSc.DM.setKSPComputeOperators>`). When PETSc coarsens a
# `DM`, it copies these callbacks from the finer `DM` to the coarser one.
# We therefore set the same callbacks on every level, and let them look
# up the level from the `DM` they are called with.


def attach_operators(dms: list[PETSc.DMShell], spaces: list[fem.FunctionSpace]) -> None:
    """Let PCMG assemble the operator of each level on its own mesh."""
    levels = {dm.handle: level_problem(V) for dm, V in zip(dms, spaces, strict=True)}

    def create_matrix(dm: PETSc.DM) -> PETSc.Mat:
        a, _ = levels[dm.handle]
        return dolfinx.fem.petsc.create_matrix(a)

    def compute_operators(ksp: PETSc.KSP, A: PETSc.Mat, P: PETSc.Mat) -> None:
        a, bc = levels[ksp.getDM().handle]
        P.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(P, a, bcs=[bc])
        P.assemble()

    for dm in dms:
        dm.setCreateMatrix(create_matrix)
        dm.setKSPComputeOperators(compute_operators)


# ### Solver
#
# We solve on the finest mesh of the hierarchy with the conjugate
# gradient method, preconditioned by full multigrid. Each level is
# smoothed by five Chebyshev iterations with a Jacobi preconditioner, and
# the coarsest level is solved with LU. The option `pc_mg_galerkin`
# selects between the Galerkin projection (`pmat`) and the rediscretised
# operators (`none`).


def solve(num_levels: int, galerkin: bool) -> tuple[int, int, float]:
    """Solve on the finest of num_levels levels.

    Returns the number of degrees-of-freedom, the number of conjugate
    gradient iterations and the L2-error.
    """
    spaces = create_hierarchy(num_levels)
    V = spaces[-1]
    msh = V.mesh

    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]) + x[0] * x[1]
    f = -ufl.div(ufl.grad(u_ex))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    u_bc = fem.Function(V, dtype=default_scalar_type)
    u_bc.interpolate(
        fem.Expression(u_ex, V.element.interpolation_points, dtype=default_scalar_type)
    )
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    boundary_dofs = fem.locate_dofs_topological(
        V, tdim - 1, mesh.exterior_facet_indices(msh.topology)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    petsc_options = {
        "ksp_type": "cg",
        "ksp_rtol": 1e-10,
        "ksp_error_if_not_converged": True,
        "pc_type": "mg",
        "pc_mg_type": "full",
        "pc_mg_levels": num_levels,
        "pc_mg_galerkin": "pmat" if galerkin else "none",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 5,
        "mg_levels_ksp_convergence_test": "skip",
        "mg_levels_pc_type": "jacobi",
    }
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix=f"demo_poisson_gmg_{num_levels}_",
    )
    dms, prolongations = attach_hierarchy(problem.solver.getPC(), spaces)
    if not galerkin:
        attach_operators(dms, spaces)
    uh = problem.solve()
    num_iterations = problem.solver.getIterationNumber()

    error = fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx, dtype=default_scalar_type)
    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error), op=MPI.SUM).real)
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    for obj in [*dms, *prolongations]:
        obj.destroy()
    problem.solver.destroy()
    return num_dofs, num_iterations, error_L2


# ## Results
#
# Adding levels refines the finest mesh. The error decreases by a factor
# of four per level, as expected for linear elements, while the number of
# conjugate gradient iterations stays small and essentially constant for
# both kinds of coarse operators. The Galerkin projection needs fewer
# iterations, as its coarse operators approximate the fine operator
# better. Rediscretisation avoids the triple matrix products, and does
# not need the fine operator as an assembled matrix.

for galerkin in (True, False):
    PETSc.Sys.Print("Galerkin projection:" if galerkin else "Rediscretisation:")
    for num_levels in range(2, 6):
        num_dofs, num_iterations, error_L2 = solve(num_levels, galerkin)
        PETSc.Sys.Print(
            f"  {num_levels} levels: {num_dofs:6d} dofs, "
            f"{num_iterations} CG iterations, L2-error {error_L2:.2e}"
        )
PETSc.garbage_cleanup(MPI.COMM_WORLD)  # type: ignore[arg-type]
