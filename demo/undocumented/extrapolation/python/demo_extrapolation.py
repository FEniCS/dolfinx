# Copyright (C) 2010 Anders Logg
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
# First added:  2010-02-08
# Last changed: 2010-02-09

from dolfin import *

# Create mesh and function spaces
mesh = UnitSquareMesh(8, 8)
P1 = FunctionSpace(mesh, "CG", 1)
P2 = FunctionSpace(mesh, "CG", 2)

# Create exact dual
dual = Expression("sin(5.0*x[0])*sin(5.0*x[1])", degree=2)

# Create P1 approximation of exact dual
z1 = Function(P1)
z1.interpolate(dual)

# Create P2 approximation from P1 approximation
z2 = Function(P2)
z2.extrapolate(z1)

# Plot approximations
plot(z1, title="z1")
plot(z2, title="z2")
interactive()
