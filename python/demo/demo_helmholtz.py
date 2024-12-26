# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Helmholtz equation
#
# Copyright (C) 2018 Samuel Groth
#
# Helmholtz problem in both complex and real modes
#
# In the complex mode, the exact solution is a plane wave propagating at
# an angle theta to the positive x-axis. Chosen for comparison with
# results from Ihlenburg's book "Finite Element Analysis of Acoustic
# Scattering" p138-139. In real mode, the exact solution corresponds to
# the real part of the plane wave (a sin function which also solves the
# homogeneous Helmholtz equation).

from mpi4py import MPI

try:
    from petsc4py import PETSc

    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)

# +
import numpy as np

import dolfinx
from dolfinx.fem import (
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
from ufl import FacetNormal, TestFunction, TrialFunction, dot, ds, dx, grad, inner

# Wavenumber
k0 = 4 * np.pi

# Approximation space polynomial degree
deg = 1

# Number of elements in each direction of the mesh
n_elem = 64

msh = create_unit_square(MPI.COMM_WORLD, n_elem, n_elem)

is_complex_mode = np.issubdtype(PETSc.ScalarType, np.complexfloating)

# Source amplitude
A = 1

# Test and trial function space
V = functionspace(msh, ("Lagrange", deg))

# +
# Function space for exact solution - need it to be higher than deg
V_exact = functionspace(msh, ("Lagrange", deg + 3))
u_exact = Function(V_exact)

# Define variational problem:
u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx

# solve for plane wave with mixed Dirichlet and Neumann BCs
theta = np.pi / 4
u_exact.interpolate(lambda x: A * np.exp(1j * k0 * (np.cos(theta) * x[0] + np.sin(theta) * x[1])))
n = FacetNormal(msh)
g = -dot(n, grad(u_exact))
L = -inner(g, v) * ds


dofs_D = locate_dofs_geometrical(
    V, lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))
)
u_bc = Function(V)
u_bc.interpolate(u_exact)
bcs = [dirichletbc(u_bc, dofs_D)]

# Compute solution
uh = Function(V)
uh.name = "u"
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    u=uh,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
problem.solve()

# Save solution in XDMF format (to be viewed in ParaView, for example)
with XDMFFile(
    MPI.COMM_WORLD, "out_helmholtz/plane_wave.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
) as file:
    file.write_mesh(msh)
    file.write_function(uh)
# -

# Calculate $L_2$ and $H^1$ errors of FEM solution and best
# approximation. This demonstrates the error bounds given in Ihlenburg.
# Pollution errors are evident for high wavenumbers.

# H1 errors
diff = uh - u_exact
H1_diff = msh.comm.allreduce(assemble_scalar(form(inner(grad(diff), grad(diff)) * dx)), op=MPI.SUM)
H1_exact = msh.comm.allreduce(
    assemble_scalar(form(inner(grad(u_exact), grad(u_exact)) * dx)), op=MPI.SUM
)
print("Relative H1 error of FEM solution:", abs(np.sqrt(H1_diff) / np.sqrt(H1_exact)))

# L2 errors
L2_diff = msh.comm.allreduce(assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
L2_exact = msh.comm.allreduce(assemble_scalar(form(inner(u_exact, u_exact) * dx)), op=MPI.SUM)
print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))
