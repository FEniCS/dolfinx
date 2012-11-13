# Copyright (C) 2012 Benjamin Kehlet
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
# First added:  2012-11-12
# Last changed: 2012-11-13

from dolfin import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)


# Define 3D geometry
box = Box(0, 0, 0, 1, 1, 1)
sphere = Sphere(Point(0, 0, 0), 0.3)
cone = Cone(Point(0, 0, -1), Point(0, 0, 1), .5, .5)

g3d = box + cone - sphere;

# Test printing
info("\nCompact output of 3D geometry:");
info(g3d);
info("\nVerbose output of 3D geometry:");
info(g3d, True);

# Plot geometry
plot(g3d, "3D geometry (surface)");

# Generate and plot mesh
mesh3d = Mesh(g3d, 128);
info(mesh3d);
plot(mesh3d, "3D mesh");

interactive();
