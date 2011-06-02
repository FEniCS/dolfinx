"""This demo colors a given mesh entities such that entities with the
same color are not neighbors. 'Neighbors' can be in the sense of shared
vertices, edges or facets, or a user-provided tuple defintion"""

# Copyright (C) 2010-2011 Garth N. Wells
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
# Modified by Anders Logg, 2010.
#
# First added:  2010-11-16
# Last changed: 2010-11-17

from dolfin import *

# Create mesh
mesh = UnitCube(24, 24, 24)

# Compute vertex-based coloring
colors = mesh.color("vertex")
plot(colors, title="Vertex-based cell coloring", interactive=True)

# Compute edge-based coloring
colors = mesh.color("edge")
plot(colors, title="Edge-based cell coloring", interactive=True)

# Compute facet-based coloring
colors = mesh.color("facet")
plot(colors, title="Facet-based cell coloring", interactive=True)

# Compute facet-based coloring with distance 2
dim = mesh.topology().dim()
coloring_type = (dim, dim - 1, dim, dim - 1, dim)
colors = mesh.color(coloring_type)
plot(colors, title="Facet-based cell coloring with distance 2", interactive=True)
