# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Geometric multigrid with matrix-free transfer operators
#
# This demo illustrates how to:
#
# - Build a hierarchy of uniformly refined meshes
# - Apply the interpolation between spaces on different meshes, and its
#   transpose, without assembling a matrix
# - Use these as the prolongation and restriction of PETSc's geometric
#   multigrid preconditioner ({py:attr}`PETSc.PC.Type.MG
#   <petsc4py.PETSc.PC.Type.MG>`), through a matrix with a Python context
#   ({py:meth}`PETSc.Mat.createPython <petsc4py.PETSc.Mat.createPython>`)
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_gmg-matrix-free-transfers.py>`
# * {download}`Jupyter notebook <./demo_gmg-matrix-free-transfers.ipynb>`
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
# ## Transfer operators
#
# Geometric multigrid solves a problem on a sequence of meshes, connected
# by a *prolongation* $P_\ell$ from level $\ell$ to level $\ell + 1$ and
# a *restriction* $R_\ell = P_\ell^T$ back. For nested Lagrange spaces,
# $P_\ell$ interpolates a coarse function into the fine space, which is
# what {py:meth}`Function.interpolate_nonmatching
# <dolfinx.fem.Function.interpolate_nonmatching>` computes. Its transpose
# is applied by {py:meth}`Function.interpolate_nonmatching_transpose
# <dolfinx.fem.Function.interpolate_nonmatching_transpose>`. Note that
# $P_\ell^T$ is not the interpolation of a fine function into the coarse
# space, which is a different operator.
#
# Both methods take the same data, which tells each interpolation point
# of the fine space which process and coarse cell to evaluate it in. For
# a mesh created by refinement, this is the parent of the fine cell
# containing the point, on the same process, and
# {py:func}`create_hierarchy_interpolation_data
# <dolfinx.fem.create_hierarchy_interpolation_data>` computes it without
# a geometric search.
#
# No matrix is stored, so the transfers need no setup beyond this data
# and no memory for matrix entries. In exchange, each application
# evaluates the coarse basis functions at the fine interpolation points
# again.
#
# ## Implementation
#
# We start by importing the required modules.

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx.fem.petsc
import dolfinx.la.petsc
import ufl
from dolfinx import default_real_type, default_scalar_type, fem, mesh

# -

# The hierarchy is a coarse mesh refined uniformly a number of times,
# with a continuous piecewise linear space on each mesh.


def create_hierarchy(
    num_levels: int,
) -> tuple[list[fem.FunctionSpace], list[np.ndarray]]:
    """Function spaces on a coarse mesh and its uniform refinements.

    The parent cell of each refined cell is returned alongside the
    spaces. The default partitioner keeps refined cells on the process
    owning their parent.
    """
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dtype=default_real_type)
    spaces = [fem.functionspace(msh, ("Lagrange", 1))]
    parent_cells = []
    for _ in range(num_levels - 1):
        msh.topology.create_entities(1)
        msh, parents, _ = mesh.refine(msh)
        # The default RefinementOption computes the parent cells
        assert parents is not None
        parent_cells.append(parents)
        spaces.append(fem.functionspace(msh, ("Lagrange", 1)))
    return spaces, parent_cells


# The problem is defined on any level of the hierarchy by the following
# functions, so that the fine problem and the operators of the coarse
# levels discretise the same problem.


# +
def exact_solution(msh: mesh.Mesh) -> ufl.core.expr.Expr:
    """The manufactured solution u_ex on the mesh msh."""
    x = ufl.SpatialCoordinate(msh)
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]) + x[0] * x[1]


def bilinear_form(V: fem.FunctionSpace) -> ufl.Form:
    """The bilinear form a(u, v) on the space V."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx


def dirichlet_bc(V: fem.FunctionSpace) -> fem.DirichletBC:
    """The Dirichlet condition u = u_ex on the boundary, in the space V."""
    msh = V.mesh
    u_bc = fem.Function(V, dtype=default_scalar_type)
    u_bc.interpolate(
        fem.Expression(
            exact_solution(msh), V.element.interpolation_points, dtype=default_scalar_type
        )
    )
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    dofs = fem.locate_dofs_topological(V, tdim - 1, mesh.exterior_facet_indices(msh.topology))
    return fem.dirichletbc(u_bc, dofs)


# -

# ### A matrix-free transfer operator
#
# A PETSc matrix of type {py:attr}`PETSc.Mat.Type.PYTHON
# <petsc4py.PETSc.Mat.Type.PYTHON>` calls the methods of a Python object
# when it is applied. The multigrid preconditioner prolongs with
# `mult` or `multAdd`, and restricts with `multTranspose` or
# `multTransposeAdd`. Each method copies the owned entries of the input
# vector into a {py:class}`Function <dolfinx.fem.Function>`, updates its
# ghost entries, applies the interpolation or its transpose, and copies
# the owned entries of the result into the output vector.


class NonmatchingTransfer:
    """Interpolation from V_coarse to V_fine, as a PETSc Python context."""

    def __init__(
        self, V_coarse: fem.FunctionSpace, V_fine: fem.FunctionSpace, parent_cells: np.ndarray
    ):
        """Work functions and interpolation data for the transfer."""
        self.u_coarse = fem.Function(V_coarse, dtype=default_scalar_type)
        self.u_fine = fem.Function(V_fine, dtype=default_scalar_type)
        tdim = V_fine.mesh.topology.dim
        num_cells = V_fine.mesh.topology.index_map(tdim).size_local
        self.cells = np.arange(num_cells, dtype=np.int32)
        self.data = fem.create_hierarchy_interpolation_data(V_fine, V_coarse, parent_cells)

    @staticmethod
    def _copy_in(x: PETSc.Vec, u: fem.Function) -> None:
        u.x.array[: x.getLocalSize()] = x.array_r
        u.x.scatter_forward()

    @staticmethod
    def _owned(u: fem.Function) -> np.ndarray:
        V = u.function_space
        return u.x.array[: V.dofmap.index_map.size_local * V.dofmap.index_map_bs]

    def mult(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Compute y = P x."""
        self._copy_in(x, self.u_coarse)
        self.u_fine.interpolate_nonmatching(self.u_coarse, self.cells, self.data)
        y.array_w[:] = self._owned(self.u_fine)

    def multTranspose(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Compute y = P^T x."""
        self._copy_in(x, self.u_fine)
        self.u_coarse.interpolate_nonmatching_transpose(self.u_fine, self.cells, self.data)
        y.array_w[:] = self._owned(self.u_coarse)

    def multAdd(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec, z: PETSc.Vec) -> None:
        """Compute z = y + P x."""
        self._copy_in(x, self.u_coarse)
        self.u_fine.interpolate_nonmatching(self.u_coarse, self.cells, self.data)
        z.array_w[:] = y.array_r + self._owned(self.u_fine)

    def multTransposeAdd(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec, z: PETSc.Vec) -> None:
        """Compute z = y + P^T x."""
        self._copy_in(x, self.u_fine)
        self.u_coarse.interpolate_nonmatching_transpose(self.u_fine, self.cells, self.data)
        z.array_w[:] = y.array_r + self._owned(self.u_coarse)


def transfer_matrix(
    V_coarse: fem.FunctionSpace, V_fine: fem.FunctionSpace, parent_cells: np.ndarray
) -> PETSc.Mat:
    """Matrix-free prolongation from V_coarse to V_fine."""

    def sizes(V: fem.FunctionSpace) -> tuple[int, int]:
        bs = V.dofmap.index_map_bs
        return (V.dofmap.index_map.size_local * bs, V.dofmap.index_map.size_global * bs)

    P = PETSc.Mat().createPython(
        (sizes(V_fine), sizes(V_coarse)),
        NonmatchingTransfer(V_coarse, V_fine, parent_cells),
        comm=V_fine.mesh.comm,  # type: ignore[arg-type]
    )
    P.setUp()
    return P


# ### The multigrid hierarchy
#
# The multigrid preconditioner builds its hierarchy by repeatedly
# coarsening the {py:class}`PETSc.DM <petsc4py.PETSc.DM>` attached to it
# with {py:meth}`PETSc.PC.setDM <petsc4py.PETSc.PC.setDM>`, and asks each
# coarse `DM` for the interpolation to the next finer one. We represent
# each level by a {py:class}`PETSc.DMShell <petsc4py.PETSc.DMShell>` with
# the parallel layout of its function space. Its coarsening, set with
# {py:meth}`setCoarsen <petsc4py.PETSc.DMShell.setCoarsen>`, returns the
# `DMShell` of the next coarser level, and its interpolation, set with
# {py:meth}`setCreateInterpolation
# <petsc4py.PETSc.DMShell.setCreateInterpolation>`, is the matrix-free
# transfer operator.
#
# A Galerkin projection $P_\ell^T A_{\ell + 1} P_\ell$ would need the
# entries of $P_\ell$, so each level instead assembles the bilinear form
# on its own mesh (rediscretisation). The preconditioner creates the
# matrix of a level with the `DM`'s matrix-creation callback
# ({py:meth}`setCreateMatrix <petsc4py.PETSc.DMShell.setCreateMatrix>`)
# and fills it with its operator callback
# ({py:meth}`setKSPComputeOperators
# <petsc4py.PETSc.DM.setKSPComputeOperators>`). When PETSc coarsens a
# `DM`, it copies these callbacks from the finer `DM` to the coarser one.
# We therefore set the same callbacks on every level, and let them look
# up the level from the `DM` they are called with. The boundary
# condition only determines which rows and columns of a level's matrix
# are constrained; its values do not enter the matrix.


# +
def shell_dm(V: fem.FunctionSpace) -> PETSc.DMShell:
    """DMShell with the parallel layout of the space V."""
    dm = PETSc.DMShell().create(V.mesh.comm)  # type: ignore[arg-type]
    dm.setGlobalVector(
        dolfinx.la.petsc.create_vector([(V.dofmap.index_map, V.dofmap.index_map_bs)])
    )
    return dm


def attach_hierarchy(
    pc: PETSc.PC,
    spaces: list[fem.FunctionSpace],
    parent_cells: list[np.ndarray],
) -> tuple[list[PETSc.DMShell], list[PETSc.Mat]]:
    """Give pc a DM hierarchy with matrix-free transfers.

    Returns the DMs and transfer operators, which must outlive the solve.
    """
    dms = [shell_dm(V) for V in spaces]
    transfers = []
    for coarse, fine, dm_c, dm_f, parents in zip(
        spaces[:-1], spaces[1:], dms[:-1], dms[1:], parent_cells, strict=True
    ):
        P = transfer_matrix(coarse, fine, parents)
        transfers.append(P)
        dm_f.setCoarsen(lambda dm, comm, dm_c=dm_c: dm_c)
        dm_c.setCreateInterpolation(lambda dmc, dmf, P=P: (P, None))  # type: ignore[arg-type]

    levels = {
        dm.handle: (fem.form(bilinear_form(V), dtype=default_scalar_type), dirichlet_bc(V))
        for dm, V in zip(dms, spaces, strict=True)
    }

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
    pc.setDM(dms[-1])
    return dms, transfers


# -

# ### Solver
#
# We solve on the finest mesh of the hierarchy with the conjugate
# gradient method, preconditioned by full multigrid. Each level is
# smoothed by five Chebyshev iterations with a Jacobi preconditioner, and
# the coarsest level is solved with LU. Setting `pc_mg_galerkin` to
# `none` selects the rediscretised operators.


def solve(num_levels: int) -> tuple[int, int, float]:
    """Solve on the finest of num_levels levels.

    Returns the number of degrees-of-freedom, the number of conjugate
    gradient iterations and the L2-error.
    """
    spaces, parent_cells = create_hierarchy(num_levels)
    V = spaces[-1]
    msh = V.mesh

    u_ex = exact_solution(msh)
    f = -ufl.div(ufl.grad(u_ex))
    a = bilinear_form(V)
    L = ufl.inner(f, ufl.TestFunction(V)) * ufl.dx
    bc = dirichlet_bc(V)

    petsc_options = {
        "ksp_type": "cg",
        "ksp_rtol": 1e-10,
        "ksp_error_if_not_converged": True,
        "pc_type": "mg",
        "pc_mg_type": "full",
        "pc_mg_levels": num_levels,
        "pc_mg_galerkin": "none",
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
        petsc_options_prefix=f"demo_gmg_matrix_free_transfers_{num_levels}_",
    )
    dms, transfers = attach_hierarchy(problem.solver.getPC(), spaces, parent_cells)
    uh = problem.solve()
    num_iterations = problem.solver.getIterationNumber()

    error = fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx, dtype=default_scalar_type)
    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error), op=MPI.SUM).real)
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    for obj in [*dms, *transfers]:
        obj.destroy()
    problem.solver.destroy()
    return num_dofs, num_iterations, error_L2


# ## Results
#
# Adding levels refines the finest mesh. The error decreases by a factor
# of four per level, as expected for linear elements, while the number of
# conjugate gradient iterations stays small and essentially constant.

for num_levels in range(2, 6):
    num_dofs, num_iterations, error_L2 = solve(num_levels)
    PETSc.Sys.Print(
        f"{num_levels} levels: {num_dofs:6d} dofs, "
        f"{num_iterations} CG iterations, L2-error {error_L2:.2e}"
    )
PETSc.garbage_cleanup(MPI.COMM_WORLD)  # type: ignore[arg-type]
