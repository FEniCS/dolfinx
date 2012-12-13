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

# Create mesh
#n = 16
#ball = Mesh(Sphere(Point(0.0, 0.0, 0.0), 1.0), n)
#mesh = BoundaryMesh(ball)
#file = File("sphere16.xml")
#file << mesh
mesh = Mesh("sphere16.xml")

# Define global normal
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

# Define function spaces and basis functions
V = FunctionSpace(mesh, "RT", 1)
Q = FunctionSpace(mesh, "DG", 0)
R = FunctionSpace(mesh, "R", 0)
W = MixedFunctionSpace((V, Q, R))

(sigma, u, r) = TrialFunctions(W)
(tau, v, t) = TestFunctions(W)

g = Expression("sin(0.5*pi*x[2])")

# Define forms
a = (inner(sigma, tau) + div(sigma)*v + div(tau)*u + r*v + t*u)*dx
L = g*v*dx

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
