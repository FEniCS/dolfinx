# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Elasticity using algebraic multigrid
#
# Copyright © 2020-2022 Garth N. Wells and Michal Habera
#
# This demo ({download}`demo_elasticity.py`) solves the equations of
# static linear elasticity using a smoothed aggregation algebraic
# multigrid solver. It illustrates how to:
#
# - Use a smoothed aggregation algebraic multigrid solver
# - Use {py:class}`Expression <dolfinx.fem.Expression>` to compute
#   derived quantities of a solution
#
# The required modules are first imported:

from mpi4py import MPI
from petsc4py import PETSc

# +
import numpy as np

import dolfinx
import ufl
from dolfinx import la
from dolfinx.fem import (
    Expression,
    Function,
    FunctionSpace,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary
from ufl import dx, grad, inner

dtype = PETSc.ScalarType  # type: ignore
# -

# ## Create the operator near-nullspace
#
# Smooth aggregation algebraic multigrid solvers require the so-called
# 'near-nullspace', which is the nullspace of the operator in the
# absence of boundary conditions. The below function builds a
# `PETSc.NullSpace` object for a 3D elasticity problem. The nullspace is
# spanned by six vectors -- three translation modes and three rotation
# modes.


def build_nullspace(V: FunctionSpace):
    """Build PETSc nullspace for 3D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    _basis = [x._cpp_object for x in basis]
    dolfinx.cpp.la.orthonormalize(_basis)
    assert dolfinx.cpp.la.is_orthonormal(_basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm)  # type: ignore
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)  # type: ignore


# ## Problem definition

# Create a box Mesh:


msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
    [16, 16, 16],
    CellType.tetrahedron,
    ghost_mode=GhostMode.shared_facet,
)

# Create a centripetal source term $f = \rho \omega^2 [x_0, \, x_1]$:

ω, ρ = 300.0, 10.0
x = ufl.SpatialCoordinate(msh)
f = ufl.as_vector((ρ * ω**2 * x[0], ρ * ω**2 * x[1], 0.0))

# Define the elasticity parameters and create a function that computes
# an expression for the stress given a displacement field.

# +
E = 1.0e9
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))


def σ(v):
    """Return an expression for the stress σ given a displacement field"""
    return 2.0 * μ * ufl.sym(grad(v)) + λ * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(len(v))


# -

# A function space space is created and the elasticity variational
# problem defined:


V = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = form(inner(σ(u), grad(v)) * dx)
L = form(inner(f, v) * dx)

# A homogeneous (zero) boundary condition is created on $x_0 = 0$ and
# $x_1 = 1$ by finding all facets on these boundaries, and then creating
# a Dirichlet boundary condition object.

facets = locate_entities_boundary(
    msh, dim=2, marker=lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0))
)
bc = dirichletbc(
    np.zeros(3, dtype=dtype), locate_dofs_topological(V, entity_dim=2, entities=facets), V=V
)

# ## Assemble and solve
#
# The bilinear form `a` is assembled into a matrix `A`, with
# modifications for the Dirichlet boundary conditions. The call
# `A.assemble()` completes any parallel communication required to
# compute the matrix.

# +
A = assemble_matrix(a, bcs=[bc])
A.assemble()
# -

# The linear form `L` is assembled into a vector `b`, and then modified
# by {py:func}`apply_lifting <dolfinx.fem.petsc.apply_lifting>` to
# account for the Dirichlet boundary conditions. After calling
# {py:func}`apply_lifting <dolfinx.fem.petsc.apply_lifting>`, the method
# `ghostUpdate` accumulates entries on the owning rank, and this is
# followed by setting the boundary values in `b`.

# +
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
set_bc(b, [bc])
# -

# Create the near-nullspace and attach it to the PETSc matrix:

ns = build_nullspace(V)
A.setNearNullSpace(ns)
A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore

# Set PETSc solver options, create a PETSc Krylov solver, and attach the
# matrix `A` to the solver:

# +
# Set solver options
opts = PETSc.Options()  # type: ignore
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-8
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10

# Create PETSc Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(msh.comm)  # type: ignore
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)
# -

# Create a solution {py:class}`Function<dolfinx.fem.Function>` `uh` and
# solve:

# +
uh = Function(V)

# Set a monitor, solve linear system, and display the solver
# configuration
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b, uh.vector)
solver.view()

# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()
# -

# ## Post-processing
#
# The computed solution is now post-processed. Expressions for the
# deviatoric and Von Mises stress are defined:

# +
sigma_dev = σ(uh) - (1 / 3) * ufl.tr(σ(uh)) * ufl.Identity(len(uh))
sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
# -

# Next, the Von Mises stress is interpolated in a piecewise-constant
# space by creating an {py:class}`Expression<dolfinx.fem.Expression>`
# that is interpolated into the
# {py:class}`Function<dolfinx.fem.Function>` `sigma_vm_h`.

# +
W = functionspace(msh, ("Discontinuous Lagrange", 0))
sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
sigma_vm_h = Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)
# -

# Save displacement field `uh` and the Von Mises stress `sigma_vm_h` in
# XDMF format files.

# +
with XDMFFile(msh.comm, "out_elasticity/displacements.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

# Save solution to XDMF format
with XDMFFile(msh.comm, "out_elasticity/von_mises_stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(sigma_vm_h)
# -

# Finally, we compute the $L^2$ norm of the displacement solution
# vector. This is a collective operation (i.e., the method `norm` must
# be called from all MPI ranks), but we print the norm only on rank 0.

# +
unorm = la.norm(uh.x)
if msh.comm.rank == 0:
    print("Solution vector norm:", unorm)
# -

# The solution vector norm can be a useful check that the solver is
# computing the same result when running in serial and in parallel.
