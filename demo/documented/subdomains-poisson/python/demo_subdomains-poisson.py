# Copyright (C) 2011 Marie E. Rognes
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
# First added:  2011-11-09
# Last changed: 2012-08-06

# Begin demo

from __future__ import print_function
from dolfin import *

# Create classes for defining parts of the boundaries and the interior
# of the domain
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (0.5, 0.7)) and between(x[0], (0.2, 1.0)))

# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()
obstacle = Obstacle()

# Define mesh
mesh = UnitSquareMesh(64, 64)

# Initialize mesh function for interior domains
domains = CellFunction("size_t", mesh)
domains.set_all(0)
obstacle.mark(domains, 1)

# Initialize mesh function for boundary domains
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

# Define input data
a0 = Constant(1.0)
a1 = Constant(0.01)
g_L = Expression("- 10*exp(- pow(x[1] - 0.5, 2))", degree=2)
g_R = Constant("1.0")
f = Constant(1.0)

# Define function space and basis functions
V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

# Define Dirichlet boundary conditions at top and bottom boundaries
bcs = [DirichletBC(V, 5.0, boundaries, 2),
       DirichletBC(V, 0.0, boundaries, 4)]

# Define new measures associated with the interior domains and
# exterior boundaries
dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Define variational form
F = (inner(a0*grad(u), grad(v))*dx(0) + inner(a1*grad(u), grad(v))*dx(1)
     - g_L*v*ds(1) - g_R*v*ds(3)
     - f*v*dx(0) - f*v*dx(1))

# Separate left and right hand sides of equation
a, L = lhs(F), rhs(F)

# Solve problem
u = Function(V)
solve(a == L, u, bcs)

# Evaluate integral of normal gradient over top boundary
n = FacetNormal(mesh)
m1 = dot(grad(u), n)*ds(2)
v1 = assemble(m1)
print("\int grad(u) * n ds(2) = ", v1)

# Evaluate integral of u over the obstacle
m2 = u*dx(1)
v2 = assemble(m2)
print("\int u dx(1) = ", v2)

# Plot solution and gradient
plot(u, title="u")
plot(grad(u), title="Projected grad(u)")
interactive()
