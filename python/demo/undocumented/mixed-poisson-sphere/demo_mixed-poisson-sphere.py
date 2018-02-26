"""This demo demonstrates how to solve a mixed Poisson type equation
defined over a sphere (the surface of a ball in 3D) including how to
create a cell_orientation map, needed for some forms defined over
manifolds."""

# Copyright (C) 2012 Marie E. Rognes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# Begin demo

from dolfin import *
import numpy
import matplotlib.pyplot as plt


# Read mesh
mesh = Mesh("../sphere_16.xml.gz")

# Define global normal
global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
mesh.init_cell_orientations(global_normal)

# Define function spaces and basis functions
RT1 = FiniteElement("RT", mesh.ufl_cell(), 1)
DG0 = FiniteElement("DG", mesh.ufl_cell(), 0)
R = FiniteElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement((RT1, DG0, R)))

(sigma, u, r) = TrialFunctions(W)
(tau, v, t) = TestFunctions(W)

g = Expression("sin(0.5*pi*x[2])", degree=2)

# Define forms
a = (inner(sigma, tau) + div(sigma)*v + div(tau)*u + r*v + t*u)*dx
L = g*v*dx

# Tune some factorization options
if has_petsc():
    # Avoid factors memory exhaustion due to excessive pivoting
    PETScOptions.set("mat_mumps_icntl_14", 40.0)
    PETScOptions.set("mat_mumps_icntl_7", "0")
    # Avoid zero pivots on 64-bit SuperLU_dist
    PETScOptions.set("mat_superlu_dist_colperm", "MMD_ATA")

# Solve problem
w = Function(W)
solve(a == L, w, solver_parameters={"symmetric": True})
(sigma, u, r) = w.split()

# Plot CG1 representation of solutions
sigma_cg = project(sigma, VectorFunctionSpace(mesh, "CG", 1))
u_cg = project(u, FunctionSpace(mesh, "CG", 1))
plt.figure()
plot(sigma_cg)
plt.figure()
plot(u_cg)
plt.show()

# Store solutions
file = File("sigma.pvd")
file << sigma
file = File("u.pvd")
file << u
