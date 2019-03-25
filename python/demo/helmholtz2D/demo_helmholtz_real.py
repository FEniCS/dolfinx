# Copyright (C) 2018 Igor Baratta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from dolfin import (MPI, Expression, FacetNormal, Function, FunctionSpace,
                    TestFunction, TrialFunction, UnitSquareMesh, dx,
                    function, grad, inner, interpolate, project, solve)
from dolfin.fem.assemble import assemble_scalar
from dolfin.io import XDMFFile


# number of elements in each direction of mesh
n_elem = 128

# approximation space polynomial degree
deg = 1

# Wavenumber
k = np.pi

mesh = UnitSquareMesh(MPI.comm_world, n_elem, n_elem)
n = FacetNormal(mesh)


# Source expression
@function.expression.numba_eval
def source(values, x, cell_idx):
    values[:, 0] = k**2 * np.cos(k * x[:, 0]) * np.cos(k * x[:, 1])


V = FunctionSpace(mesh, ("Lagrange", deg))

# Prepare Expression as FE function
f = interpolate(Expression(source), V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx - k**2 * inner(u, v) * dx
L = inner(f, v) * dx

# Compute solution
u = Function(V)
solve(a == L, u, [])

# Save solution in XDMF format (to be viewed in Paraview, for example)
with XDMFFile(MPI.comm_world, "plane_wave.xdmf",
              encoding=XDMFFile.Encoding.HDF5) as file:
    file.write(u)


"""Calculate L2 and H1 errors of FEM solution and best approximation."""


@function.expression.numba_eval
def solution(values, x, cell_idx):
    values[:, 0] = np.cos(k * x[:, 0]) * np.cos(k * x[:, 1])


# Function space for exact solution - need it to be higher than deg
V_exact = FunctionSpace(mesh, ("Lagrange", deg + 3))

# "exact" solution
u_exact = interpolate(Expression(solution), V_exact)

# best approximation from V
u_BA = project(u_exact, V)

# H1 errors
diff = u - u_exact
diff_BA = u_BA - u_exact
H1_diff = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(diff), grad(diff)) * dx))
H1_BA = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(diff_BA), grad(diff_BA)) * dx))
H1_exact = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_exact), grad(u_exact)) * dx))
print("Relative H1 error of best approximation:", np.sqrt(H1_BA) / np.sqrt(H1_exact))
print("Relative H1 error of FEM solution:", np.sqrt(H1_diff) / np.sqrt(H1_exact))

# L2 errors
L2_diff = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(diff, diff) * dx))
L2_BA = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(diff_BA, diff_BA) * dx))
L2_exact = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(u_exact, u_exact) * dx))
print("Relative L2 error  of best approximation:", np.sqrt(L2_BA) / np.sqrt(L2_exact))
print("Relative L2 error of FEM solution:", np.sqrt(L2_diff) / np.sqrt(L2_exact))
