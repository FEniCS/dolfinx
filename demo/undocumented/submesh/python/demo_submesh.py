"""This demo program demonstrates how to extract matching sub meshes
from a common mesh."""

# Copyright (C) 2009, 2015 Anders Logg
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
# First added:  2009-02-11
# Last changed: 2015-06-15

from __future__ import print_function
from dolfin import *

# Structure sub domain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.4 - DOLFIN_EPS and x[0] < 1.6 \
            + DOLFIN_EPS and x[1] < 0.6 + DOLFIN_EPS

# Create mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(3.0, 1.0), 60, 20)

# Create sub domain markers and mark everaything as 0
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim())
sub_domains.set_all(0)

# Mark structure domain as 1
structure = Structure()
structure.mark(sub_domains, 1)

# Extract sub meshes
fluid_mesh = SubMesh(mesh, sub_domains, 0)
structure_mesh = SubMesh(mesh, sub_domains, 1)

# Move structure mesh
for x in structure_mesh.coordinates():
    x[0] += 0.1*x[0]*x[1]

# Move fluid mesh according to structure mesh
ALE.move(fluid_mesh, structure_mesh)
fluid_mesh.smooth()

# Plot meshes
plot(fluid_mesh, title="Fluid")
plot(structure_mesh, title="Structure")
interactive()

# Build mapping from structure to fluid mesh
structure_to_fluid = compute_vertex_map(structure_mesh, fluid_mesh)
print(structure_to_fluid)
