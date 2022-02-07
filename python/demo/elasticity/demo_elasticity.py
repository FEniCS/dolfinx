#
# .. _demo_elasticity:
#
# Elasticity equation
# ===================
# Copyright (C) 2020 Garth N. Wells and Michal Habera
#
# This demo solves the equations of static linear elasticity. The solver
# uses smoothed aggregation algebraic multigrid. ::

from contextlib import ExitStack

import numpy as np

from dolfinx import la
from dolfinx.fem import (Expression, Function, FunctionSpace,
                         VectorFunctionSpace, apply_lifting, assemble_matrix,
                         assemble_vector, dirichletbc, form,
                         locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, dx, grad, inner, sqrt, sym, tr)

from mpi4py import MPI
from petsc4py import PETSc

# Nullspace and problem setup
# ---------------------------
#
# Prepare a helper which builds a PETSc NullSpace. Nullspace (or near
# nullspace) is needed to improve the performance of algebraic
# multigrid.
#
# In the case of small deformation linear elasticity the nullspace
# contains rigid body modes. ::


def build_nullspace(V):
    """Function to build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for null space
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for i in range(6)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]

        # Build translational nullspace basis
        for i in range(3):
            basis[i][dofs[i]] = 1.0

        # Build rotational nullspace basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        basis[3][dofs[0]] = -x1
        basis[3][dofs[1]] = x0
        basis[4][dofs[0]] = x2
        basis[4][dofs[2]] = -x0
        basis[5][dofs[2]] = x1
        basis[5][dofs[1]] = -x2

    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)
    return PETSc.NullSpace().create(vectors=ns)


mesh = create_box(
    MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                     np.array([2.0, 1.0, 1.0])], [12, 12, 12],
    CellType.tetrahedron, GhostMode.shared_facet)


# Rotation rate and mass density
omega = 300.0
rho = 10.0

# Loading due to centripetal acceleration (rho*omega^2*x_i)
x = SpatialCoordinate(mesh)
f = as_vector((rho * omega**2 * x[0], rho * omega**2 * x[1], 0.0))

# Elasticity parameters
E = 1.0e9
nu = 0.0
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def sigma(v):
    return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(
        len(v))


# Create function space
V = VectorFunctionSpace(mesh, ("Lagrange", 1))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = form(inner(sigma(u), grad(v)) * dx)
L = form(inner(f, v) * dx)

# Set up boundary condition on inner surface
bc = dirichletbc(np.array([0, 0, 0], dtype=PETSc.ScalarType),
                 locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                    np.isclose(x[1], 1.0))), V)

# Assembly and solve
# ------------------
# ::

# Assemble system, applying boundary conditions
A = assemble_matrix(a, bcs=[bc])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Create solution function
u = Function(V)

# Create near null space basis (required for smoothed aggregation AMG)
null_space = build_nullspace(V)

# Attach near nullspace to matrix
A.setNearNullSpace(null_space)

# Set solver options
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-12
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

# Create CG Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)

# Compute solution
solver.setMonitor(lambda ksp, its, rnorm: print("Iteration: {}, rel. residual: {}".format(its, rnorm)))
solver.solve(b, u.vector)
solver.view()
u.x.scatter_forward()

# Compute von Mises stress via interpolation
sigma_deviatoric = sigma(u) - (1 / 3) * tr(sigma(u)) * Identity(len(u))
sigma_von_mises = sqrt((3 / 2) * inner(sigma_deviatoric, sigma_deviatoric))

W = FunctionSpace(mesh, ("Discontinuous Lagrange", 0))
sigma_von_mises_expression = Expression(sigma_von_mises, W.element.interpolation_points)
sigma_von_mises_h = Function(W)
sigma_von_mises_h.interpolate(sigma_von_mises_expression)

# Save solution to XDMF format
with XDMFFile(MPI.COMM_WORLD, "displacements.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)

# Save solution to XDMF format
with XDMFFile(MPI.COMM_WORLD, "von_mises_stress.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(sigma_von_mises_h)

unorm = u.x.norm()
if mesh.comm.rank == 0:
    print("Solution vector norm:", unorm)
