"""This demo demonstrates how to move the vertex coordinates of a
boundary mesh and then updating the interior vertex coordinates of the
original mesh by suitably interpolating the vertex coordinates (useful
for implementation of ALE methods)."""

# Copyright (C) 2008 Solveig Bruvoll and Anders Logg
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
# First added:  2008-05-02
# Last changed: 2008-12-12

from dolfin import *

# Create mesh
mesh = UnitSquareMesh(20, 20)

# Create boundary mesh
boundary = BoundaryMesh(mesh, "exterior")

# Move vertices in boundary
for x in boundary.coordinates():
    x[0] *= 3.0
    x[1] += 0.1*sin(5.0*x[0])

# Move mesh
ALE.move(mesh, boundary)

# Plot mesh
plot(mesh, interactive=True)

# Write mesh to file
File("deformed_mesh.pvd") << mesh
