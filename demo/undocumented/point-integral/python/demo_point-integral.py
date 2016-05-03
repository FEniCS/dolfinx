"""This demo program solves Poisson's equation with a delta dirac point source

    - div grad u(x, y) = f(x, y)*dirac

on the unit square with source f given by

    f(x, y) = 0.4

and boundary conditions given by

    u(x, y) = 0          for x = 0 or x = 1
du/dn(x, y) = A*sin(5*x) for y = 0 or y = 1
"""

# Copyright (C) 2014 Johan Hake
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
# First added:  2014-10-20
# Last changed: 2014-10-20

# Begin demo

from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

def center_func(x):
    return (0.45 <= x[0] and x[0] <= 0.55 and near(x[1], 0.5)) or \
           0.45 <= x[1] and x[1] <= 0.55 and near(x[0], 0.5)

# Define domain for point integral
center_domain = VertexFunction("size_t", mesh, 0)
center = AutoSubDomain(center_func)
center.mark(center_domain, 1)
dPP = dP(subdomain_data=center_domain)

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.4)
g = Expression("A*sin(5*x[0])", A=10.0, degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dPP(1) + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
plot(u, interactive=True)
