"""This demo program solves Poisson's equation

    - div grad u(x) = f(x)

on the unit interval with source f given by

    f(x) = 9*pi^2*sin(3*pi*x[0])

and boundary conditions given by

    u(x) = 0 for x = 0
    du/dx = 0 for x = 1
"""

# Copyright (C) 2007 Kristian B. Oelgaard
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
# Modified by Anders Logg 2011
#
# First added:  2007-11-28
# Last changed: 2012-11-12

from dolfin import *

# Create mesh and function space
mesh = UnitIntervalMesh(50)
V = FunctionSpace(mesh, "CG", 1)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("9.0*pi*pi*sin(3.0*pi*x[0])", degree=2)
g = Expression("3.0*pi*cos(3.0*pi*x[0])", degree=2)

a = dot(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution to file
file = File("poisson.pvd")
file << u

# Plot solution
plot(u, interactive=True)
