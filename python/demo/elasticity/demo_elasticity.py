# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This demo solves the equations of static linear elasticity for a
# pulley subjected to centripetal accelerations. The solver uses
# smoothed aggregation algebraic multigrid.

from contextlib import ExitStack

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import BoxMesh, DirichletBC, Function, VectorFunctionSpace, cpp
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (Form, apply_lifting, assemble_matrix, assemble_vector,
                         locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.la import VectorSpaceBasis
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, dx, grad, inner, sym, tr)


def build_nullspace(V):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    index_map = V.dofmap.index_map
    nullspace_basis = [cpp.la.create_vector(index_map) for i in range(6)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        for i in range(3):
            basis[i][V.sub(i).dofmap.list.array] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]
        basis[3][dofs[0]] = -x[dofs[0], 1]
        basis[3][dofs[1]] = x[dofs[1], 0]
        basis[4][dofs[0]] = x[dofs[0], 2]
        basis[4][dofs[2]] = -x[dofs[2], 0]
        basis[5][dofs[2]] = x[dofs[2], 1]
        basis[5][dofs[1]] = -x[dofs[1], 2]

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(6)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp


# Load mesh from file
# mesh = Mesh(MPI.COMM_WORLD)
# XDMFFile(MPI.COMM_WORLD, "../pulley.xdmf").read(mesh)

# mesh = UnitCubeMesh(2, 2, 2)
mesh = BoxMesh(
    MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                     np.array([2.0, 1.0, 1.0])], [12, 12, 12],
    CellType.tetrahedron, dolfinx.cpp.mesh.GhostMode.none)

# Function to mark inner surface of pulley
# def inner_surface(x, on_boundary):
#    r = 3.75 - x[2]*0.17
#    return (x[0]*x[0] + x[1]*x[1]) < r*r and on_boundary


def boundary(x):
    return np.logical_or(x[0] < 10.0 * np.finfo(float).eps,
                         x[0] > 1.0 - 10.0 * np.finfo(float).eps)


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

# Stress computation


def sigma(v):
    return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(
        len(v))


# Create function space
V = VectorFunctionSpace(mesh, ("Lagrange", 1))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v)) * dx
L = inner(f, v) * dx

u0 = Function(V)
with u0.vector.localForm() as bc_local:
    bc_local.set(0.0)

# Set up boundary condition on inner surface
bc = DirichletBC(u0, locate_dofs_geometrical(V, boundary))

# Explicitly compile a UFL Form into dolfinx Form
form = Form(a, jit_parameters={"cffi_extra_compile_args": "-Ofast -march=native", "cffi_verbose": True})

# Assemble system, applying boundary conditions and preserving symmetry
A = assemble_matrix(form, [bc])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Create solution function
u = Function(V)

# Create near null space basis (required for smoothed aggregation AMG).
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

# Save solution to XDMF format
with XDMFFile(MPI.COMM_WORLD, "elasticity.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)

unorm = u.vector.norm()
if mesh.mpi_comm().rank == 0:
    print("Solution vector norm:", unorm)
