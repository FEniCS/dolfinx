"""This script demonstrate how to project and interpolate functions
between different finite element spaces."""

# Copyright (C) 2008 Anders Logg
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
# First added:  2008-10-06
# Last changed: 2012-11-12

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh and define function spaces
mesh = UnitSquareMesh(64, 64)
P1 = FunctionSpace(mesh, "CG", 1)

# Define function
v = Expression("sin(10.0*x[0])*sin(10.0*x[1])", degree=2)

# Compute projection (L2-projection)
Pv = project(v, V=P1)

# Compute interpolation (evaluating dofs)
PIv = Function(P1)
PIv.interpolate(v)

# Plot functions
plt.figure()
plot(v, mesh=mesh, title="v")
plt.figure()
plot(Pv,  title="Pv")
plt.figure()
plot(PIv, title="PI v")
plt.show()
