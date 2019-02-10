# Copyright (C) 2018 Samuel Groth
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

u"""Test Helmholtz problem for which the exact solution is a plane wave
propagating at angle theta to the postive x-axis. Chosen for
comparison with results from Ihlenburg\'s book \"Finite Element
Analysis of Acoustic Scattering\" p138-139"""

import numpy as np

from dolfin import (MPI, Expression, FacetNormal, Function, FunctionSpace,
                    TestFunction, TrialFunction, UnitSquareMesh, dot, ds, dx,
                    function, grad, has_petsc_complex, inner, interpolate,
                    project, solve)
from dolfin.fem.assemble import assemble_scalar
from dolfin.io import XDMFFile

if not has_petsc_complex:
    print('This demo only works with PETSc-complex')
    exit()


# Wavenumber
k0 = 20

# approximation space polynomial degree
deg = 1

# number of elements in each direction of mesh
n_elem = 128

mesh = UnitSquareMesh(MPI.comm_world, n_elem, n_elem)
n = FacetNormal(mesh)

# Incident plane wave
theta = np.pi / 8


@function.expression.numba_eval
def ui_eval(values, x, cell_idx):
    values[:, 0] = np.exp(1.0j * k0 * (np.cos(theta) * x[:, 0] + np.sin(theta) * x[:, 1]))


# Test and trial function space
V = FunctionSpace(mesh, ("Lagrange", deg))

# Prepare Expression as FE function
ui = interpolate(Expression(ui_eval), V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
g = dot(grad(ui), n) + 1j * k0 * ui
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx + 1j * k0 * inner(u, v) * ds
L = inner(g, v) * ds

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
# Function space for exact solution - need it to be higher than deg
V_exact = FunctionSpace(mesh, ("Lagrange", deg + 3))

# "exact" solution
u_exact = interpolate(Expression(ui_eval), V_exact)

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
