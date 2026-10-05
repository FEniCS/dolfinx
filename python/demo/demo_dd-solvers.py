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
from dolfinx import fem, la, mesh
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

# Both solvers report how many of their constrained degrees of freedom
# lie on a subdomain interface. Filling a vector with ones and
# accumulating it onto the owning process gives, for each degree of
# freedom, the number of processes that hold it.


def num_shared_dofs(V: fem.FunctionSpace, dofs: np.ndarray) -> int:
    """Count the entries of ``dofs`` held by more than one process.

    Args:
        V: Space the degrees of freedom belong to.
        dofs: Degrees of freedom to test, as local indices.

    Returns:
        How many of ``dofs`` this process shares with another.
    """
    sharers = la.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.int32)
    sharers.array[:] = 1
    sharers.scatter_reverse(la.InsertMode.add)
    sharers.scatter_forward()
    return int(np.count_nonzero(sharers.array[dofs] > 1))


# `solve_poisson` builds the Poisson problem on a given mesh and solves
# it with conjugate gradients preconditioned by BDDC.


def solve_poisson(msh: mesh.Mesh) -> tuple[fem.Function, int, int]:
    """Solve the Poisson problem on ``msh`` using BDDC.

    Args:
        msh: Mesh, which must have been built without ghost cells.

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
    problem = LinearProblem(
        a,
        L,
        u=uh,
        bcs=bcs,
        kind="is",
        petsc_options_prefix=f"demo_dd_poisson_{V.dofmap.index_map.size_global}_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "bddc",
            "ksp_rtol": 1e-5 if np.finfo(dtype).bits == 32 else 1e-8,
            "ksp_error_if_not_converged": True,
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


def rigid_body_modes(V: fem.FunctionSpace) -> PETSc.NullSpace:
    """Build the rigid body modes of a 2D displacement space.

    Args:
        V: Vector-valued displacement space.

    Returns:
        The two translations and one rotation, orthonormalised.
    """
    bs = V.dofmap.index_map_bs
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for _ in range(3)]
    b = [mode.array for mode in basis]
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(2)]

    # Two translations
    b[0][dofs[0]] = 1.0
    b[1][dofs[1]] = 1.0

    # One rotation, about the origin
    x = V.tabulate_dof_coordinates()
    blocks = V.dofmap.list.flatten()
    b[2][dofs[0]] = -x[blocks, 1]
    b[2][dofs[1]] = x[blocks, 0]

    la.orthonormalize(basis)

    # Copied into PETSc vectors rather than wrapped, so that the null
    # space does not outlive the arrays it was built from
    num_owned = bs * V.dofmap.index_map.size_local
    vectors = []
    for mode in b:
        vec = PETSc.Vec().createMPI((num_owned, None), bsize=bs, comm=V.mesh.comm)  # type: ignore[arg-type]
        vec.array_w[:] = mode[:num_owned]
        vectors.append(vec)
    return PETSc.NullSpace().create(vectors=vectors)


# `solve_elasticity` assembles the operator and the right-hand side
# itself, rather than through
# :class:`~dolfinx.fem.petsc.LinearProblem`, so that the near null
# space can be attached to the matrix and the solver configured
# directly.


def solve_elasticity(msh: mesh.Mesh) -> tuple[fem.Function, int, int]:
    """Solve the elasticity problem on ``msh`` using BDDC.

    Args:
        msh: Mesh, which must have been built without ghost cells.

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

    # Assemble the unassembled (MATIS) operator and the right-hand side
    A = assemble_matrix(a, bcs=[bc], kind="is")
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
    set_bc(b, [bc])

    near_nullspace = rigid_body_modes(V)
    A.setNearNullSpace(near_nullspace)
    A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore[arg-type]

    ksp = PETSc.KSP().create(msh.comm)  # type: ignore[arg-type]
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setTolerances(rtol=1e-5 if np.finfo(dtype).bits == 32 else 1e-8, max_it=100)
    ksp.getPC().setType("bddc")

    # Read once into the PC, then removed so that the two solves in the
    # loop below do not inherit each other's settings
    opts = PETSc.Options()
    opts["pc_bddc_use_change_of_basis"] = True  # type: ignore[index]
    ksp.setFromOptions()
    del opts["pc_bddc_use_change_of_basis"]  # type: ignore[arg-type]

    uh = fem.Function(V, name="u", dtype=dtype)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    if ksp.getConvergedReason() < 0:
        raise RuntimeError(f"Elasticity solve failed: {ksp.getConvergedReason()}")

    its = ksp.getIterationNumber()
    num_shared = num_shared_dofs(V, dofs)
    ksp.destroy()
    A.destroy()
    b.destroy()

    # The modes hold the vectors they were built from, so releasing
    # them here keeps the solve free of residue
    near_nullspace.destroy()
    return uh, its, num_shared


# The number of BDDC iterations is close to independent of the mesh
# size, so refining the mesh does not slow convergence the way it would
# for a one-level method. Solving on a sequence of meshes shows this.
# Each mesh is built once and handed to both solvers.

# +
comm = MPI.COMM_WORLD
for n in (32, 64):
    # BDDC requires one non-overlapping subdomain per process, so the
    # mesh is built without ghost cells
    msh = mesh.create_unit_square(
        comm, n, n, mesh.CellType.triangle, ghost_mode=mesh.GhostMode.none, dtype=xdtype
    )

    uh, its, shared = solve_poisson(msh)
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    error = fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx, dtype=dtype)
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM).real)
    num_dofs = uh.function_space.dofmap.index_map.size_global
    shared = comm.allreduce(shared, MPI.SUM)
    if comm.rank == 0:
        print(
            f"Poisson,    n = {n:>3d}: {num_dofs:>7d} dofs, {its:>3d} CG iterations, "
            f"L2 error = {l2_error:.3e}, shared Dirichlet dofs = {shared}"
        )

    uh, its, shared = solve_elasticity(msh)
    V = uh.function_space
    energy = fem.form(0.5 * ufl.inner(uh, uh) * ufl.dx, dtype=dtype)
    norm = np.sqrt(comm.allreduce(fem.assemble_scalar(energy), MPI.SUM).real)
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    shared = comm.allreduce(shared, MPI.SUM)
    if comm.rank == 0:
        print(
            f"Elasticity, n = {n:>3d}: {num_dofs:>7d} dofs, {its:>3d} CG iterations, "
            f"|u|_L2 = {norm:.3e}, shared Dirichlet dofs = {shared}"
        )
# -

# With more than one process, `shared Dirichlet dofs` is non-zero: those
# are the constrained vertices where a subdomain interface meets the
# Dirichlet boundary, and they are the reason the boundary condition
# value has to be distributed across the processes that share them
# rather than written by the owner alone. The Poisson problem is
# constrained on the whole boundary and the elasticity problem on one
# edge, so the counts differ.
