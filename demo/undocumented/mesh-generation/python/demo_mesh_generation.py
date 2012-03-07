# Copyright (C) 2012 Garth N. Wells
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
# First added:  2012-02-02
# Last changed:
#
# This program generates a mesh for a polygonal domain that is
# represented by a list of its vertices.

from dolfin import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Create empty Mesh
mesh = Mesh()

# Create list of polygonal domain vertices
domain_vertices = [Point(0.0, 0.0),
                   Point(10.0, 0.0),
                   Point(10.0, 2.0),
                   Point(8.0, 2.0),
                   Point(7.5, 1.0),
                   Point(2.5, 1.0),
                   Point(2.0, 4.0),
                   Point(0.0, 4.0),
                   Point(0.0, 0.0)]

# Generate mesh and plot
PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.25);
plot(mesh, interactive=True)

# Polyhedron face vertices
face_vertices = [Point(0.0, 0.0, 0.0),
                 Point(0.0, 0.0, 1.0),
                 Point(0.0, 1.0, 0.0),
                 Point(1.0, 0.0, 0.0)]

# Polyhedron faces (of a tetrahedron)
face0 = [3, 2, 1]
face1 = [0, 3, 1]
face2 = [0, 2, 3]
face3 = [0, 1, 2]
faces = [face0, face1, face2, face3]

# Generate 3D mesh and plot
PolyhedralMeshGenerator.generate(mesh, face_vertices, faces, 0.05)
plot(mesh, interactive=True)

# Generate 3D mesh from OFF file input (cube)
PolyhedralMeshGenerator.generate(mesh, "../cube.off", 0.05)
plot(mesh, interactive=True)
