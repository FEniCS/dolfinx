# Copyright (C) 2012-2013 Benjamin Kehlet
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
# Last changed: 2013-03-15

from dolfin import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)


# Define 2D geometry
r = Rectangle(0.5, 0.5, 1.5, 1.5);
c = Circle (1, 1, 1);
g2d = c - r;

# Test printing
info("\nCompact output of 2D geometry:");
info(g2d);
info("");
info("\nVerbose output of 2D geometry:");
info(g2d, True);

# Plot geometry
plot(g2d, "2D Geometry (boundary)");

# Generate and plot mesh
mesh2d = Mesh(g2d, 10);
plot(mesh2d, "2D mesh");

interactive();
