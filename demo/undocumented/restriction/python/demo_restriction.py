"""This demo program solves a Poisson problem on a function space that lives on a subset of the mesh."""

# Copyright (C) 2013 Martin S. Alnaes
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
# First added:  2013-04-03
# Last changed: 2013-04-03

from dolfin import *

# Note that the interface to restricted function spaces is experimental
# and likely to change somewhat in future dolfin versions.

class Domain (SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] > 0.25 - DOLFIN_EPS and
                x[0] < 0.75 + DOLFIN_EPS and
                x[1] > 0.25 - DOLFIN_EPS and
                x[1] < 0.75 + DOLFIN_EPS)

class Boundary(SubDomain):
   def inside(self, x, on_boundary):
       return (abs(x[0] - 0.25) < DOLFIN_EPS and
               x[1] > 0.25 - DOLFIN_EPS and
               x[1] < 0.75 + DOLFIN_EPS)

mesh = UnitSquareMesh(32, 32)
domain = Domain()

markers = CellFunction("size_t", mesh)
markers.set_all(1)
domain.mark(markers, 0)

dx = dx[markers]

# Two ways of defining a Restriction
#restriction = Restriction(mesh, domain)
restriction = Restriction(markers, 0)

V = FunctionSpace(restriction, "CG", 1)

# Define functions
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(100.0)

# Define forms
a = dot(grad(u),grad(v))*dx()
L = f*v*dx()

# Define boundary condition
zero = Constant(0.0)
boundary = Boundary()
bc = DirichletBC(V, zero, boundary)

# Solve system
u = Function(V)
solve(a == L, u, bc)

# Plot solution, note that the restricted function is zero-extended, this may change in future versions
plot(u)
interactive()
