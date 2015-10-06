"""This demo illustrates the built-in mesh types."""

# Copyright (C) 2008 Garth N. Wells
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
# Modified by Anders Logg 2008, 2014, 2015
# Modified by Benjamin Kehlet 2012
#
# Begin demo

from __future__ import print_function
from dolfin import *

# Check for debugging of buildbot failures
import os
if MPI.size(mpi_comm_world()) > 1 and not bool(int(os.environ.get("DOLFIN_NOPLOT", "0"))):
    raise RuntimeError("When running this demo in parallel, DOLFIN_NOPLOT must be set to disable the plot windows.")

mesh = UnitIntervalMesh(10)
print("Plotting a UnitIntervalMesh")
plot(mesh, title="Unit interval")

mesh = UnitSquareMesh(10, 10)
print("Plotting a UnitSquareMesh")
plot(mesh, title="Unit square")

mesh = UnitSquareMesh(10, 10, "left")
print("Plotting a UnitSquareMesh")
plot(mesh, title="Unit square (left)")

mesh = UnitSquareMesh(10, 10, "crossed")
print("Plotting a UnitSquareMesh")
plot(mesh, title="Unit square (crossed)")

mesh = UnitSquareMesh(10, 10, "right/left")
print("Plotting a UnitSquareMesh")
plot(mesh, title="Unit square (right/left)")

mesh = RectangleMesh(Point(0.0, 0.0), Point(10.0, 4.0), 10, 10)
print("Plotting a RectangleMesh")
plot(mesh, title="Rectangle")

mesh = RectangleMesh(Point(-3.0, 2.0), Point(7.0, 6.0), 10, 10, "right/left")
print("Plotting a RectangleMesh")
plot(mesh, title="Rectangle (right/left)")

mesh = UnitCubeMesh(10, 10, 10)
print("Plotting a UnitCubeMesh")
plot(mesh, title="Unit cube")

mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(10.0, 4.0, 2.0), 10, 10, 10)
print("Plotting a BoxMesh")
plot(mesh, title="Box")

interactive()
