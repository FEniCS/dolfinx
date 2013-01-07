"""This script demonstrates how to interpolate functions between different
finite element spaces on non-matching meshes."""

# Copyright (C) 2009 Garth N. Wells
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
# First added:  2009-05-19
# Last changed: 2009-05-19

from dolfin import *

# Create mesh and define function spaces
mesh0 = UnitSquareMesh(16, 16)
mesh1 = UnitSquareMesh(64, 64)

P1 = FunctionSpace(mesh1, "CG", 1)
P3 = FunctionSpace(mesh0, "CG", 3)

# Define function
v0 = Expression("sin(10.0*x[0])*sin(10.0*x[1])", element=FiniteElement('CG', triangle, 3))
v1 = Function(P1)

# Interpolate
v1.interpolate(v0)

# Plot functions
plot(v0, mesh=mesh0, title="v0")
plot(v1, title="v1")
interactive()

print norm(v0, mesh = mesh1)
print norm(v1)
