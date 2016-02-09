"""This demo demonstrates how to solve a mixed Poisson type equation
defined over a sphere (the surface of a ball in 3D) including how to
create a cell_orientation map, needed for some forms defined over
manifolds."""

# Copyright (C) 2012 Marie E. Rognes
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2012-12-09
# Last changed: 2012-12-09

# Begin demo

from dolfin import *
import numpy

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

# Set PETSc MUMPS parameters (this is required to prevent a memory
# error when using MUMPS LU solver, which is probably due to the Real
# space).
if has_petsc():
    PETScOptions.set("mat_mumps_icntl_14", 40.0)
    PETScOptions.set("mat_mumps_icntl_7", "0")

# Solve problem
w = Function(W)
solve(a == L, w)
(sigma, u, r) = w.split()

# Plot CG1 representation of solutions
sigma_cg = project(sigma, VectorFunctionSpace(mesh, "CG", 1))
u_cg = project(u, FunctionSpace(mesh, "CG", 1))
plot(sigma_cg, interactive=True)
plot(u_cg)

# Store solutions
file = File("sigma.pvd")
file << sigma
file = File("u.pvd")
file << u

interactive()
