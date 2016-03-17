"""This demo illustrates how to use of DOLFIN for solving a nonlinear
PDE, in this case a nonlinear variant of Poisson's equation,

    - div (1 + u^2) grad u(x, y) = f(x, y)

on the unit square with source f given by

     f(x, y) = x*sin(y)

and boundary conditions given by

     u(x, y)     = 1  for x = 0
     du/dn(x, y) = 0  otherwise

This is equivalent to solving the variational problem

    F(u) = ((1 + u^2)*grad(u), grad(v)) - (f, v) = 0

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
# Original implementation (../cpp/main.cpp) by Garth N. Wells.
#
# Modified by Anders Logg, 2008.
# Modified by Harish Narayanan, 2009.
#
# First added:  2007-11-14
# Last changed: 2013-11-20
# Begin demo

from dolfin import *

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
File("mesh.pvd") << mesh

V = FunctionSpace(mesh, "CG", 1)

# Define boundary condition
g = Constant(1.0)
bc = DirichletBC(V, g, DirichletBoundary())

# Define variational problem
u = Function(V)
v = TestFunction(V)
f = Expression("x[0]*sin(x[1])", degree=2)
F = inner((1 + u**2)*grad(u), grad(v))*dx - f*v*dx

# Compute solution
solve(F == 0, u, bc, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})

# Plot solution and solution gradient
plot(u, title="Solution")
plot(grad(u), title="Solution gradient")
interactive()

# Save solution in VTK format
file = File("nonlinear_poisson.pvd")
file << u
