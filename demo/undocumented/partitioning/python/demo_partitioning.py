"""Run this demo in parallel by

    mpirun -n <n> python demo.py

where <n> is the desired number of processes."""

# Copyright (C) 2008 Ola Skavhaug and Anders Logg
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
# First added:  2008-12-18
# Last changed: 2008-12-18

from dolfin import *

# Read in mesh from XML file in parallel
mesh = Mesh("unitsquare.xml.gz")

# Plot partition
plot(mesh, interactive=True)
