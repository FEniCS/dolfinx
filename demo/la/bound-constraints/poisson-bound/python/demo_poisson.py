"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1

under the bound constraints 

	0 <= u(x, y) <= x
	
This is an exemple of how to use An example of use of TAO 
to solve bound constrained  problems
"""

# Copyright (C) 2007-2011 Anders Logg
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
# First added:  16/04/2012
# Last changed: 16/04/2012

# Begin demo
# An example of use of the interface to TAO to solve bound constrained  problems in FEnics:
# This is a poisson solver with bound constraints 
#
# Corrado Maurini 
#
from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)
# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Assemble the linear system
A=PETScMatrix()
b=PETScVector()
A=assemble(a)
b=assemble(L)
bc.apply(A)
bc.apply(b)

# Define the upper and lower bounds
upperbound = interpolate(Expression("x[1]"), V) # example of non-uniform upper-bound
lowerbound = interpolate(Constant(0.), V) # example of a uniform lower-bound
xu=upperbound.vector() # or xu=down_cast(upperbound.vector())
xl=lowerbound.vector() # or xl=down_cast(lowerbound.vector())

# Define the function to store the solution and the related PETScVector
usol=Function(V);
xsol=usol.vector() # or xsol=down_cast(usol.vector())

# Create the TAOLinearBoundSolver and solve the problem
solver=TAOLinearBoundSolver()
solver.solve(A,xsol,b,xl,xu)
help(solver)


# Save solution in VTK format
#file = File("poisson.pvd")
#file << usol
# Plot solution
plot(usol, interactive=True)
