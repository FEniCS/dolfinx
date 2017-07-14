"""This script demonstrates the L2 projection of a function onto a
non-matching mesh."""

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
# Modified by Anders Logg 2011
#
# First added:  2009-10-10
# Last changed: 2012-11-12

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh and define function spaces
mesh0 = UnitSquareMesh(16, 16)
mesh1 = UnitSquareMesh(64, 64)

# Create expression on P3
u0 = Expression("sin(10.0*x[0])*sin(10.0*x[1])", degree=3)

# Define projection space
P1 = FunctionSpace(mesh1, "CG", 1)

# Define projection variation problem
v  = TestFunction(P1)
u1 = TrialFunction(P1)
a  = v*u1*dx
L  = v*u0*dx

# Compute solution
u1 = Function(P1)
solve(a == L, u1)

# Plot functions
plt.figure()
plot(u0, mesh=mesh0, title="u0")
plt.figure()
plot(u1, title="u1")
plt.show()
