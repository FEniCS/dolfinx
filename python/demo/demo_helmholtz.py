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
# In the complex mode, the exact solution is a plane wave propagating at
# an angle theta to the positive x-axis. Chosen for comparison with
# results from Ihlenburg's book "Finite Element Analysis of Acoustic
# Scattering" p138-139. In real mode, the Method of Manufactured
# Solutions is used to produce the exact solution and source term.

from mpi4py import MPI
from petsc4py import PETSc

# +
import numpy as np

import ufl
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner

# Wavenumber
k0 = 4 * np.pi

# Approximation space polynomial degree
deg = 1

# Number of elements in each direction of the mesh
n_elem = 128

msh = create_unit_square(MPI.COMM_WORLD, n_elem, n_elem)

# Source amplitude
if np.issubdtype(PETSc.ScalarType, np.complexfloating):  # type: ignore
    A = PETSc.ScalarType(1 + 1j)  # type: ignore
else:
    A = 1

# Test and trial function space
V = functionspace(msh, ("Lagrange", deg))

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = Function(V)
f.interpolate(lambda x: A * k0**2 * np.cos(k0 * x[0]) * np.cos(k0 * x[1]))
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx
L = inner(f, v) * dx

# Compute solution
uh = Function(V)
uh.name = "u"
problem = LinearProblem(
    a,
    L,
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

# +
# Function space for exact solution - need it to be higher than deg
V_exact = functionspace(msh, ("Lagrange", deg + 3))
u_exact = Function(V_exact)
u_exact.interpolate(lambda x: A * np.cos(k0 * x[0]) * np.cos(k0 * x[1]))

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
