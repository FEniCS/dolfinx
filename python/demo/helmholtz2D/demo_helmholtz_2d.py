# Copyright (C) 2018 Samuel Groth
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

u"""Test Helmholtz problem in both complex and real modes
In the complex mode, the exact solution is a plane wave propagating at an angle
theta to the positive x-axis. Chosen for comparison with results from Ihlenburg\'s
book \"Finite Element Analysis of Acoustic Scattering\" p138-139.
In real mode, the Method of Manufactured Solutions is used to produce the exact
solution and source term."""


import numpy as np

from dolfin import (MPI, Expression, FacetNormal, Function, FunctionSpace,
                    TestFunction, TrialFunction, UnitSquareMesh, dot, ds, dx,
                    function, grad, has_petsc_complex, inner, interpolate,
                    project, solve)
from dolfin.fem.assemble import assemble_scalar
from dolfin.io import XDMFFile


# Wavenumber
k0 = 4 * np.pi

# approximation space polynomial degree
deg = 1

# number of elements in each direction of mesh
n_elem = 128

mesh = UnitSquareMesh(MPI.comm_world, n_elem, n_elem)
n = FacetNormal(mesh)


if has_petsc_complex:
    # Incident plane wave direction
    theta = np.pi / 8

    @function.expression.numba_eval
    def source(values, x, cell_idx):
        values[:, 0] = np.exp(1.0j * k0 * (np.cos(theta) * x[:, 0] + np.sin(theta) * x[:, 1]))
else:
    @function.expression.numba_eval
    def source(values, x, cell_idx):
        values[:, 0] = k0**2 * np.cos(k0 * x[:, 0]) * np.cos(k0 * x[:, 1])


# Test and trial function space
V = FunctionSpace(mesh, ("Lagrange", deg))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
if has_petsc_complex:
    ui = interpolate(Expression(source), V)
    g = dot(grad(ui), n) + 1j * k0 * ui
    a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx +\
        1j * k0 * inner(u, v) * ds
    L = inner(g, v) * ds
else:
    f = interpolate(Expression(source), V)
    a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx
    L = inner(f, v) * dx

# Compute solution
u = Function(V)
solve(a == L, u, [])

# Save solution in XDMF format (to be viewed in Paraview, for example)
with XDMFFile(MPI.comm_world, "plane_wave.xdmf",
              encoding=XDMFFile.Encoding.HDF5) as file:
    file.write(u)

"""Calculate L2 and H1 errors of FEM solution and best approximation.
This demonstrates the error bounds given in Ihlenburg. Pollution errors
are evident for high wavenumbers."""

# "Exact" solution expression
if has_petsc_complex:
    @function.expression.numba_eval
    def solution(values, x, cell_idx):
        values[:, 0] = np.exp(1.0j * k0 * (np.cos(theta) * x[:, 0] + np.sin(theta) * x[:, 1]))
else:
    @function.expression.numba_eval
    def solution(values, x, cell_idx):
        values[:, 0] = np.cos(k0 * x[:, 0]) * np.cos(k0 * x[:, 1])


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
