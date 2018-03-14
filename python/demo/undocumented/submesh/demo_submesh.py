"""This demo program demonstrates how to extract matching sub meshes
from a common mesh."""

# Copyright (C) 2009, 2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *
import matplotlib.pyplot as plt

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
plt.figure()
plot(fluid_mesh, title="Fluid")
plt.figure()
plot(structure_mesh, title="Structure")
plt.show()

# Build mapping from structure to fluid mesh
structure_to_fluid = compute_vertex_map(structure_mesh, fluid_mesh)
print(structure_to_fluid)
