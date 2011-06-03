"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)

and boundary conditions given by

    u(x, y)     = 0
    du/dn(x, y) = 0

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
#
# Modified by Anders Logg, 2008.
#
# First added:  2007-10-02
# Last changed: 2008-12-19

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(24, 24)
V = FunctionSpace(mesh, "DG", 1)

# Define test and trial functions
v = TestFunction(V)
u = TrialFunction(V)

# Define normal component, mesh size and right-hand side
n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2
f = Expression("500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

# Define parameters
alpha = 4.0
gamma = 8.0

w = Function(V)
F = dot(grad(v), grad(w))*dx \
   - dot(avg(grad(v)), jump(w, n))*dS \
   - dot(jump(v, n), avg(grad(w)))*dS \
   + alpha/h_avg*dot(jump(v, n), jump(w, n))*dS \
   - dot(grad(v), w*n)*ds \
   - dot(v*n, grad(w))*ds \
   + gamma/h*v*w*ds

# Define bilinear form
a = dot(grad(v), grad(u))*dx \
   - dot(avg(grad(v)), jump(u, n))*dS \
   - dot(jump(v, n), avg(grad(u)))*dS \
   + alpha/h_avg*dot(jump(v, n), jump(u, n))*dS \
   - dot(grad(v), u*n)*ds \
   - dot(v*n, grad(u))*ds \
   + gamma/h*v*u*ds

a = derivative(F, w, u)

# Define linear form
L = v*f*dx

# Compute solution
problem = VariationalProblem(a, L)
u = problem.solve()
print "norm:", u.vector().norm("l2")

# Project solution to piecewise linears
u_proj = project(u)

# Save solution to file
file = File("poisson.pvd")
file << u_proj

# Plot solution
plot(u_proj, interactive=True)
