# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation: ../cpp/main.cpp by Anders Logg
#
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
# Modified by Mikael Mortensen 2012
#
# First added:  2007-11-15
# Last changed: 2012-11-8

from dolfin import *

# Create mesh and finite element
mesh = UnitSquareMesh(32, 32)

# Source term
class Source(Expression):
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) + 1.0*exp(-(dx*dx + dy*dy)/0.02)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) and on_boundary)

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

class PeriodicBoundaryRight(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(abs(x[0]-1.) < DOLFIN_EPS and on_boundary)

pbc = PeriodicBoundary()
pbr = PeriodicBoundaryRight()
mf = FacetFunction('sizet', mesh)
mf.set_all(0)
pbc.mark(mf, 1)
pbr.mark(mf, 2)
#mesh.domains().markers(1).assign(mf)
mesh.add_periodic_direction(pbc)
#mesh.add_periodic_direction(1, 2)

V = FunctionSpace(mesh, "CG", 2)

# Create Dirichlet boundary condition
u0 = Constant(0.0)
dbc = DirichletBoundary()
bc0 = DirichletBC(V, u0, dbc)

# Collect boundary conditions
bcs = [bc0]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

f = Source()
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Save solution to file
file = File("periodic_dofmap.xml.gz")
file << u

#u = Function(V, "periodic_dofmap.xml.gz")
list_timings()

# Plot solution
plot(u, interactive=True)



