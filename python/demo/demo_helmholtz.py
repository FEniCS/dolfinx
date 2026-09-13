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
# Copyright (C) 2018-2025 Samuel Groth and Jørgen S. Dokken
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_helmholtz.py>`
# * {download}`Jupyter notebook <./demo_helmholtz.ipynb>`
# ```
# This demo illustrates how to:
# - Create a complex-valued finite element formulation
# In the following example, we will consider the Helmholtz equation solved
# with both a complex valued and a real valued finite element formulation.
#
# In the complex mode, the exact solution is a plane wave propagating at
# an angle theta to the positive x-axis. Chosen for comparison with
# results from Ihlenburg's book [Finite Element Analysis of Acoustic
# Scattering, p138-139](https://doi.org/10.1007/0-387-22700-8_4).
# In real mode, the exact solution corresponds to
# the real part of the plane wave (a sin function which also solves the
# homogeneous Helmholtz equation).

# We start by importing the necessary modules

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx.fem import (
    Expression,
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

# -

# We define the necessary parameters for the discretized problem

k0 = 4 * np.pi  # Wavenumber
deg = 1  # Approximation space polynomial degree
n_elem = 64  # Number of elements in each direction of the mesh
A = 1  # Source amplitude

# Next, we create the discrete domain, a unit square and set up the
# discrete function space.

msh = create_unit_square(MPI.COMM_WORLD, n_elem, n_elem)
V = functionspace(msh, ("Lagrange", deg))

# ## Define variational problem.
# The Helmholtz equation can be discretized in the same way for both the
# real and complex valued formulation. However, note that we use
# `ufl.inner` instead of `ufl.dot` or the ` * ` operator between
# the test and trial function, and that the test-function is
# **always** the second variable in the operator.
# The reason for this is that for complex variational forms,
# one requires a sesquilinear two-form, with the inner product being
# $(a,b)=\int_\Omega a \cdot \bar{b}~\mathrm{d}x$.

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k0**2 * ufl.inner(u, v) * ufl.dx

# We solve for plane wave with mixed Dirichlet and Neumann BCs.
# We use ufl to manufacture an exact solution and corresponding
# boundary conditions.


# +
theta = np.pi / 4
V_exact = functionspace(
    msh, ("Lagrange", deg + 3)
)  # Exact solution should be in a higher order space
u_exact = Function(V_exact, name="u_exact")
u_exact.interpolate(lambda x: A * np.exp(1j * k0 * (np.cos(theta) * x[0] + np.sin(theta) * x[1])))
x = ufl.SpatialCoordinate(msh)
n = ufl.FacetNormal(msh)
g = -ufl.dot(n, ufl.grad(u_exact))
L = -ufl.inner(g, v) * ufl.ds

dofs_D = locate_dofs_geometrical(
    V, lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[1], 0))
)
u_bc = Function(V)
u_bc.interpolate(Expression(u_exact, V.element.interpolation_points))
bcs = [dirichletbc(u_bc, dofs_D)]
# -

# In this problem, we rely on PETSc as the linear algebra backend.
# PETSc can only be configured for a either real of complex valued matrices
# and vectors. We can check how PETSc is configured by calling

# +
is_complex_mode = np.issubdtype(PETSc.ScalarType, np.complexfloating)
PETSc.Sys.Print(f"PETSc is configured in complex mode: {is_complex_mode}")

uh = Function(V)
uh.name = "u"
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    u=uh,
    petsc_options_prefix="demo_helmholtz_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)
_ = problem.solve()
# -

# Save solution in XDMF format (to be viewed in ParaView, for example)

with XDMFFile(
    MPI.COMM_WORLD, "out_helmholtz/plane_wave.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
) as file:
    file.write_mesh(msh)
    file.write_function(uh)

# Calculate $L_2$ and $H_0^1$ errors of FEM solution and best
# approximation. This demonstrates the error bounds given in Ihlenburg.
# Pollution errors are evident for high wavenumbers.

diff = uh - u_exact
H1_diff = msh.comm.allreduce(
    assemble_scalar(form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx)), op=MPI.SUM
)
H1_exact = msh.comm.allreduce(
    assemble_scalar(form(ufl.inner(ufl.grad(u_exact), ufl.grad(u_exact)) * ufl.dx)), op=MPI.SUM
)
PETSc.Sys.Print("Relative H1 error of FEM solution:", abs(np.sqrt(H1_diff) / np.sqrt(H1_exact)))

L2_diff = msh.comm.allreduce(assemble_scalar(form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM)
L2_exact = msh.comm.allreduce(
    assemble_scalar(form(ufl.inner(u_exact, u_exact) * ufl.dx)), op=MPI.SUM
)
PETSc.Sys.Print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))
