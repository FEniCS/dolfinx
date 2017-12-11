"""This program is used to generate the coefficients c00, c01 and c11
used in the demo."""

# Copyright (C) 2007-2009 Anders Logg
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
# First added:  2009-12-16
# Last changed: 2009-12-16
# Begin demo

from dolfin import *

# Create mesh
mesh = UnitSquareMesh(32, 32)

# Create mesh functions for c00, c01, c11
c00 = MeshFunction("double", mesh, 2)
c01 = MeshFunction("double", mesh, 2)
c11 = MeshFunction("double", mesh, 2)

# Iterate over mesh and set values
for cell in cells(mesh):
    if cell.midpoint().x() < 0.5:
        c00[cell] = 1.0
        c01[cell] = 0.3
        c11[cell] = 2.0
    else:
        c00[cell] = 3.0
        c01[cell] = 0.5
        c11[cell] = 4.0

# Store to file
mesh_file = File("../unitsquare_32_32.xml.gz")
c00_file = File("../unitsquare_32_32_c00.xml.gz")
c01_file = File("../unitsquare_32_32_c01.xml.gz")
c11_file = File("../unitsquare_32_32_c11.xml.gz")

mesh_file << mesh
c00_file << c00
c01_file << c01
c11_file << c11

# Plot mesh functions
plot(c00, title="C00", interactive=True)
plot(c01, title="C01", interactive=True)
plot(c11, title="C11", interactive=True)
