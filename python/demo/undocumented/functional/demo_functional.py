"""This demo program computes the value of the functional

    M(v) = int v^2 + (grad v)^2 dx

on the unit square for v = sin(x) + cos(y). The exact
value of the functional is M(v) = 2 + 2*sin(1)*(1 - cos(1))

The functional M corresponds to the energy norm for a
simple reaction-diffusion equation."""

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
# First added:  2007-11-14
# Last changed: 2012-11-12


from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 2)

# Define the function v
v = Expression("sin(x[0]) + cos(x[1])",
               element=V.ufl_element(),
               domain=mesh)

# Define functional
M = (v*v + dot(grad(v), grad(v)))*dx(mesh)

# Evaluate functional
value = assemble(M)

exact_value = 2.0 + 2.0*sin(1.0)*(1.0 - cos(1.0))
print("The energy norm of v is: %.15g" % value)
print("It should be:            %.15g" % exact_value)
