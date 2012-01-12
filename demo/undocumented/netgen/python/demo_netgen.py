"This demo program creates a 3D mesh of a standard LEGO brick."

# Copyright (C) 2012 Anders Logg
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
# First added:  2012-01-12
# Last changed: 2012-01-12

from dolfin import *
from dolfin.mesh.netgen import *

# Create an interesting geometry (adapted from NETGEN demo cubeandspheres.geo)
cube = Brick(0, 0, 0, 1, 1, 1)
inner = Sphere(0.5, 0.5, 0.5, 0.58)
outer = Sphere(0.5, 0.5, 0.5, 0.75)
cutcube = cube*outer - inner

# Generate and plot mesh
cutcube_mesh = Mesh(cutcube, meshsize="fine")
plot(cutcube_mesh)

# Create an 8 x 2 LEGO brick
lego = LEGO(8, 2, 3)

# Generate and plot mesh
lego_mesh = Mesh(lego)
plot(lego_mesh)

# It is also possible to explicitly write geometry and mesh files
lego.write_geometry("lego.geo")
lego.generate_mesh("lego.xml")

# Hold plots
interactive()
