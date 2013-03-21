"""This demo illustrates use of the MeshFunction class.

Original implementation: ../cpp/main.cpp by Ola Skavhaug."""

# Copyright (C) 2007 Kristian B. Oelgaard
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
# First added:  2007-11-15
# Last changed: 2008-03-31

from dolfin import *

# Read mesh from file
mesh = Mesh("../mesh2D.xml.gz")

# Read mesh function from file
file_in = File("../meshfunction.xml.gz")
f = MeshFunction("double", mesh)
file_in >> f

# Write mesh function to file
out = File("meshfunction_out.xml.gz");
out << f

# Plot mesh function
plot(f, interactive=True)
