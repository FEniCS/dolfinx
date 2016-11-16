"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = -100*exp(-((x - 0.5)^2 + (y - 0.5)^2)/0.02)

and boundary conditions given by

    u(x, y)     = u0 on x = 0 and x = 1
    du/dn(x, y) = g  on y = 0 and y = 1

where

    u0 = x + 0.25*sin(2*pi*x)
    g = (y - 0.5)**2

using a discontinuous Galerkin formulation (interior penalty method).
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

from dolfin import *

# FIXME: Make mesh ghosted
parameters["ghost_mode"] = "shared_facet"

# Define class marking Dirichlet boundary (x = 0 or x = 1)
class DirichletBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0]*(1 - x[0]), 0)

# Define class marking Neumann boundary (y = 0 or y = 1)
class NeumanBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[1]*(1 - x[1]), 0)

# Create mesh and define function space
mesh = UnitSquareMesh(24, 24)
V = FunctionSpace(mesh, 'DG', 1)

# Define test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal vector and mesh size
n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2

# Define the source term f, Dirichlet term u0 and Neumann term g
f = Expression('-100.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)', degree=2)
u0 = Expression('x[0] + 0.25*sin(2*pi*x[1])', degree=2)
g = Expression('(x[1] - 0.5)*(x[1] - 0.5)', degree=2)

# Mark facets of the mesh
boundaries = FacetFunction('size_t', mesh, 0)
NeumanBoundary().mark(boundaries, 2)
DirichletBoundary().mark(boundaries, 1)

# Define outer surface measure aware of Dirichlet and Neumann boundaries
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Define parameters
alpha = 4.0
gamma = 8.0

# Define variational problem
a = dot(grad(v), grad(u))*dx \
   - dot(avg(grad(v)), jump(u, n))*dS \
   - dot(jump(v, n), avg(grad(u)))*dS \
   + alpha/h_avg*dot(jump(v, n), jump(u, n))*dS \
   - dot(grad(v), u*n)*ds(1) \
   - dot(v*n, grad(u))*ds(1) \
   + (gamma/h)*v*u*ds(1)
L = v*f*dx - u0*dot(grad(v), n)*ds(1) + (gamma/h)*u0*v*ds(1) + g*v*ds(2)

# Compute solution
u = Function(V)
solve(a == L, u)
print("Solution vector norm (0): {!r}".format(u.vector().norm("l2")))

# Save solution to file
#file = File("poisson.pvd")
#file << u

# Plot solution
#plot(u, interactive=True)
