"""This demo program solves the reaction-diffusion equation

    - div grad u + u = f

on the unit square with f = sin(x)*sin(y) and homogeneous Neumann
boundary conditions.

This demo is also available in more compact form in short.py,
the world's maybe shortest PDE solver.
"""

# Copyright (C) 2009-2011 Anders Logg
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
# First added:  2009-06-15
# Last changed: 2012-11-12

from dolfin import *

# Define variational problem
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("sin(x[0])*sin(x[1])")
a = dot(grad(u), grad(v))*dx + u*v*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u)

# Plo solution
plot(u, interactive=True)
