# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This demo solves the equations of static linear elasticity for a
# pulley subjected to centripetal accelerations. The solver uses
# smoothed aggregation algerbaric multigrid.

import numpy as np
from petsc4py import PETSc

import dolfin
from dolfin import (MPI, BoxMesh, CellType, DirichletBC, Function, Point,
                    TestFunction, TrialFunction, VectorFunctionSpace, cpp)
from dolfin.fem.assembling import assemble_system
from dolfin.io import XDMFFile
from dolfin.la import PETScKrylovSolver, PETScOptions, VectorSpaceBasis
from ufl import Identity, as_vector, dx, grad, inner, sym, tr


def build_nullspace(V):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    index_map = V.dofmap().index_map()
    nullspace_basis = [cpp.la.create_vector(index_map) for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0)

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[3], 1.0, 0)
    V.sub(0).set_x(nullspace_basis[4], 1.0, 2)
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0)
    V.sub(2).set_x(nullspace_basis[5], 1.0, 1)
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2)

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(6)]
    nsp = PETSc.NullSpace()
    nsp.create(_x)
    return nsp


# Load mesh from file
# mesh = Mesh(MPI.comm_world)
# XDMFFile(MPI.comm_world, "../pulley.xdmf").read(mesh)

# mesh = UnitCubeMesh(2, 2, 2)
mesh = BoxMesh.create(
    MPI.comm_world, [Point(0, 0, 0)._cpp_object,
                     Point(2, 1, 1)._cpp_object], [12, 12, 12],
    CellType.Type.tetrahedron, dolfin.cpp.mesh.GhostMode.none)
cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
mesh.geometry.coord_mapping = cmap

# Function to mark inner surface of pulley
# def inner_surface(x, on_boundary):
#    r = 3.75 - x[2]*0.17
#    return (x[0]*x[0] + x[1]*x[1]) < r*r and on_boundary


def boundary(x, on_boundary):
    return np.logical_or(x[:, 0] < np.finfo(float).eps,
                         x[:, 0] > 1.0 - np.finfo(float).eps)


# Rotation rate and mass density
omega = 300.0
rho = 10.0

# Loading due to centripetal acceleration (rho*omega^2*x_i)
# f = Expression(("rho*omega*omega*x[0]", "rho*omega*omega*x[1]", "0.0"),

f = as_vector((0.0, 1.0E+10, 0.0))

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
# Set up boundary condition on inner surface
bc = DirichletBC(V, u0, boundary)

# Assemble system, applying boundary conditions and preserving symmetry)
A, b = assemble_system(a, L, bc)
assert A.block_size == 3

# Create solution function
u = Function(V)

# Create near null space basis (required for smoothed aggregation AMG).
null_space = build_nullspace(V)

# Attach near nullspace to matrix
A.setNearNullSpace(null_space)

# Set solver options
PETScOptions.set("ksp_view")
PETScOptions.set("ksp_type", "cg")
PETScOptions.set("ksp_rtol", 1.0e-12)
PETScOptions.set("pc_type", "gamg")

# Use Chebyshev smoothing for multigrid
PETScOptions.set("mg_levels_ksp_type", "chebyshev")
PETScOptions.set("mg_levels_pc_type", "jacobi")

# Improve estimate of eigenvalues for Chebyshev smoothing
PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 20)

# Monitor solver
PETScOptions.set("ksp_monitor")

# Create CG Krylov solver and turn convergence monitoring on
solver = PETScKrylovSolver(MPI.comm_world)
solver.set_from_options()

# Set matrix operator
solver.set_operator(A)

# Compute solution
solver.solve(u.vector(), b)

# Save solution to XDMF format
file = XDMFFile(MPI.comm_world, "elasticity.xdmf")
file.write(u)

unorm = u.vector().norm()
if MPI.rank(mesh.mpi_comm()) == 0:
    print("Solution vector norm:", unorm)

# Save colored mesh partitions in VTK format if running in parallel
# if MPI.size(mesh.mpi_comm()) > 1:
#    File("partitions.pvd") << MeshFunction("size_t", mesh, mesh.topology.dim, \
#                                           MPI.rank(mesh.mpi_comm()))

# Project and write stress field to post-processing file
# W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
# stress = project(sigma(u), V=W)
# File("stress.pvd") << stress

# Plot solution
# import matplotlib.pyplot as plt
# import dolfin.plotting
# dolfin.plotting.plot(u)
# plt.show()
