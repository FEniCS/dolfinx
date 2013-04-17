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
domain = Rectangle(0., 0., 5., 5.) - Rectangle(2., 1.25, 3., 1.75) - Circle(1, 4, .25) - Circle(4, 4, .25)
domain.subdomain(Rectangle(1., 1., 4., 3.))
domain.subdomain(Rectangle(2., 2., 3., 4.))


# Test printing
info("\nCompact output of 2D geometry:");
info(domain);
info("");
info("\nVerbose output of 2D geometry:");
info(domain, True);

# Plot geometry
plot(domain, "2D Geometry (boundary)");

# Generate and plot mesh
mesh2d = Mesh(domain, 45);

f = File("outmesh.xml")
f << mesh2d

print "Mesh generated", mesh2d
plot(mesh2d.domains().cell_domains(), "Subdomains");
plot(mesh2d, "2D mesh")

interactive();
