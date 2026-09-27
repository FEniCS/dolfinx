# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Poisson equation with a BDDC preconditioner
#
# This demo illustrates how to:
#
# - Assemble an operator in PETSc's unassembled `MATIS` format.
# - Solve it with a Balancing Domain Decomposition by Constraints
#   (BDDC) preconditioner.
# - Set up a mesh so that each MPI process owns one non-overlapping
#   subdomain.
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_poisson-bddc.py>`
# * {download}`Jupyter notebook <./demo_poisson-bddc.ipynb>`
# ```
#
# ## Equation and problem definition
#
# We solve the Poisson equation on the unit square with homogeneous
# Dirichlet conditions on the whole boundary,
#
# $$
# \begin{aligned}
#   - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
#   u &= 0 \quad {\rm on} \ \partial \Omega,
# \end{aligned}
# $$
#
# and take $f = 2 \pi^{2} \sin(\pi x_{0}) \sin(\pi x_{1})$, for which
# the exact solution is $u = \sin(\pi x_{0}) \sin(\pi x_{1})$. The
# variational problem is: find $u \in V$ such that
#
# $$
# \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x
# = \int_{\Omega} f v \, {\rm d} x \quad \forall \ v \in V,
# $$
#
# with $V$ the space of piecewise linear Lagrange functions.
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
# an interface meets the boundary the vertex there is held by every
# process meeting at that point. Each of them must place a value on its
# local diagonal, or $A_{i}$ is again singular — but those values are
# summed, so each contributes a share rather than the whole value. The
# demo reports how many such degrees of freedom there are.
#
# ## Implementation

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem.petsc import LinearProblem

dtype = PETSc.ScalarType
xdtype = PETSc.RealType
# -

# `solve` builds the problem on an `n` by `n` mesh and solves it with
# conjugate gradients preconditioned by BDDC. It returns the computed
# solution, the iteration count, and the number of Dirichlet degrees of
# freedom this process shares with a neighbour.


# +
def solve(n: int) -> tuple[fem.Function, int, int]:
    """Solve the Poisson problem on an ``n`` by ``n`` mesh using BDDC.

    Args:
        n: Number of cells in each direction.

    Returns:
        The solution, the number of Krylov iterations, and the number
        of Dirichlet degrees of freedom shared with another process.
    """
    # BDDC requires one non-overlapping subdomain per process, so the
    # mesh is built without ghost cells
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, mesh.CellType.triangle, ghost_mode=mesh.GhostMode.none, dtype=xdtype
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 2 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    boundary_dofs = fem.locate_dofs_topological(
        V, tdim - 1, mesh.exterior_facet_indices(msh.topology)
    )
    bcs = [fem.dirichletbc(dtype(0), boundary_dofs, V)]

    # Count the Dirichlet degrees of freedom that lie on a subdomain
    # interface. Filling a vector with ones and accumulating it onto
    # the owning process gives, for each degree of freedom, the number
    # of processes that hold it
    sharers = la.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.int32)
    sharers.array[:] = 1
    sharers.scatter_reverse(la.InsertMode.add)
    sharers.scatter_forward()
    num_shared = int(np.count_nonzero(sharers.array[boundary_dofs] > 1))

    uh = fem.Function(V, name="u", dtype=dtype)
    problem = LinearProblem(
        a,
        L,
        u=uh,
        bcs=bcs,
        kind="is",
        petsc_options_prefix=f"demo_poisson_bddc_{n}_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "bddc",
            "ksp_rtol": 1e-8,
            "ksp_error_if_not_converged": True,
        },
    )
    problem.solve()
    return uh, problem.solver.getIterationNumber(), num_shared


# -

# The number of BDDC iterations is close to independent of the mesh
# size, so refining the mesh does not slow convergence the way it would
# for a one-level method. Solving on a sequence of meshes shows this,
# and lets us confirm that the error decreases at the expected rate.

# +
comm = MPI.COMM_WORLD
for n in (32, 64):
    uh, its, num_shared = solve(n)

    V = uh.function_space
    msh = V.mesh
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    error = fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx, dtype=dtype)
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM).real)

    shared = comm.allreduce(num_shared, MPI.SUM)
    num_dofs = V.dofmap.index_map.size_global
    PETSc.Sys.Print(
        f"n = {n:>3d}: {num_dofs:>6d} dofs, {its:>3d} CG iterations, "
        f"L2 error = {l2_error:.3e}, shared Dirichlet dofs = {shared}"
    )
# -

# With more than one process, `shared Dirichlet dofs` is non-zero: those
# are the boundary vertices where a subdomain interface meets
# $\partial \Omega$, and they are the reason the boundary condition
# value has to be distributed across the processes that share them
# rather than written by the owner alone.
