# Copyright (C) 2018 Samuel Groth
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

u"""Test Helmholtz problem in both complex and real modes
In the complex mode, the exact solution is a plane wave propagating at an angle
theta to the positive x-axis. Chosen for comparison with results from Ihlenburg\'s
book \"Finite Element Analysis of Acoustic Scattering\" p138-139.
In real mode, the Method of Manufactured Solutions is used to produce the exact
solution and source term."""

import numpy as np
from mpi4py import MPI

from dolfinx import (Function, FunctionSpace, UnitSquareMesh,
                     has_petsc_complex, solve)
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.io import XDMFFile
from ufl import FacetNormal, TestFunction, TrialFunction, dx, grad, inner

# wavenumber
k0 = 4 * np.pi

# approximation space polynomial degree
deg = 1

# number of elements in each direction of mesh
n_elem = 128

mesh = UnitSquareMesh(MPI.COMM_WORLD, n_elem, n_elem)
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
f.interpolate(lambda x: A * k0**2 * np.cos(k0 * x[0]) * np.cos(k0 * x[1]))
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx
L = inner(f, v) * dx

# Compute solution
u = Function(V)
solve(a == L, u, [])

# Save solution in XDMF format (to be viewed in Paraview, for example)
with XDMFFile(MPI.COMM_WORLD, "plane_wave.xdmf", "w",
              encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)
    file.write_function(u)

"""Calculate L2 and H1 errors of FEM solution and best approximation.
This demonstrates the error bounds given in Ihlenburg. Pollution errors
are evident for high wavenumbers."""


# "Exact" solution expression
def solution(values, x):
    values[:, 0] = A * np.cos(k0 * x[:, 0]) * np.cos(k0 * x[:, 1])


# Function space for exact solution - need it to be higher than deg
V_exact = FunctionSpace(mesh, ("Lagrange", deg + 3))
u_exact = Function(V_exact)
u_exact.interpolate(lambda x: A * np.cos(k0 * x[0]) * np.cos(k0 * x[1]))

# best approximation from V
# u_BA = project(u_exact, V)

# H1 errors
diff = u - u_exact
H1_diff = mesh.mpi_comm().allreduce(assemble_scalar(inner(grad(diff), grad(diff)) * dx), op=MPI.SUM)
H1_exact = mesh.mpi_comm().allreduce(assemble_scalar(inner(grad(u_exact), grad(u_exact)) * dx), op=MPI.SUM)
print("Relative H1 error of FEM solution:", abs(np.sqrt(H1_diff) / np.sqrt(H1_exact)))

# L2 errors
L2_diff = mesh.mpi_comm().allreduce(assemble_scalar(inner(diff, diff) * dx), op=MPI.SUM)
L2_exact = mesh.mpi_comm().allreduce(assemble_scalar(inner(u_exact, u_exact) * dx), op=MPI.SUM)
print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))
