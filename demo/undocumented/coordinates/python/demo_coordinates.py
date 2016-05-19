"""This demo program demonstrates how to manipulate (higher-order) mesh
coordinates."""

# Copyright (C) 2016 Jan Blechta
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

from dolfin import *

# Create mesh
comm = mpi_comm_world()
mesh = UnitDiscMesh(comm, 20, 2, 2)
plot(mesh, interactive=True)

# Fetch coordinate function
C = FunctionSpace(mesh, mesh.ufl_coordinate_element())
c = Function(C)
get_coordinates(c, mesh.geometry())

# Deform coordinates harmonically subject to BC
u, v = TrialFunction(C), TestFunction(C)
a = inner(grad(u), grad(v))*dx
L = dot(Constant((0, 0)), v)*dx
bc1 = DirichletBC(C, (-1, -1), "x[0] < -0.5")
bc2 = DirichletBC(C, c, "x[0] >= -0.5")
displacement = Function(C)
solve(a == L, displacement, [bc1, bc2])
c_vec = c.vector()
c_vec += displacement.vector()

# Set coordinates
set_coordinates(mesh.geometry(), c)
plot(mesh, interactive=True)
