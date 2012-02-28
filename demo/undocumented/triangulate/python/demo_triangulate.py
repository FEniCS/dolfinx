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
# This demo creates meshes from the triangulation of a collection of
# random point.

from dolfin import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Create list of random points
num_points = 2000
random_points = [Point(dolfin.rand(), dolfin.rand(), dolfin.rand()) for p in range(num_points)]

# Create empty Mesh
mesh = Mesh()

# Triangulate points in 2D and plot mesh
Triangulate.triangulate(mesh, random_points, 2)
plot(mesh, interactive=True)

# Triangulate points in 3D and plot mesh
Triangulate.triangulate(mesh, random_points, 3)
plot(mesh, interactive=True)
