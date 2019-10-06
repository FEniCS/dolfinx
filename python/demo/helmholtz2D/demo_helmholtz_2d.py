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

from dolfin import (MPI, FacetNormal, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, has_petsc_complex,
                    interpolate, solve)
from dolfin.fem.assemble import assemble_scalar
from dolfin.io import XDMFFile
from ufl import dx, grad, inner

# wavenumber
k0 = 4 * np.pi

# approximation space polynomial degree
deg = 1

# number of elements in each direction of mesh
n_elem = 128

mesh = UnitSquareMesh(MPI.comm_world, n_elem, n_elem)
n = FacetNormal(mesh)

# Source amplitude
if has_petsc_complex:
    A = 1 + 1j
else:
    A = 1


# Test and trial function space
V = FunctionSpace(mesh, ("Lagrange", deg))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
f.interpolate(lambda x: A * k0**2 * np.cos(k0 * x[:, 0]) * np.cos(k0 * x[:, 1]))
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
def solution(values, x):
    values[:, 0] = A * np.cos(k0 * x[:, 0]) * np.cos(k0 * x[:, 1])


# Function space for exact solution - need it to be higher than deg
V_exact = FunctionSpace(mesh, ("Lagrange", deg + 3))
u_exact = Function(V_exact)
u_exact.interpolate(lambda x: A * np.cos(k0 * x[:, 0]) * np.cos(k0 * x[:, 1]))

# best approximation from V
# u_BA = project(u_exact, V)

# H1 errors
diff = u - u_exact
# diff_BA = u_BA - u_exact
H1_diff = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(diff), grad(diff)) * dx))
# H1_BA = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(diff_BA), grad(diff_BA)) * dx))
H1_exact = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_exact), grad(u_exact)) * dx))
# print("Relative H1 error of best approximation:", abs(np.sqrt(H1_BA) / np.sqrt(H1_exact)))
print("Relative H1 error of FEM solution:", abs(np.sqrt(H1_diff) / np.sqrt(H1_exact)))

# L2 errors
L2_diff = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(diff, diff) * dx))
# L2_BA = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(diff_BA, diff_BA) * dx))
L2_exact = MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(u_exact, u_exact) * dx))
# print("Relative L2 error  of best approximation:", abs(np.sqrt(L2_BA) / np.sqrt(L2_exact)))
print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))
