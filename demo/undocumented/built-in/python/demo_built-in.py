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
# Modified by Anders Logg, 2008.
# Modified by Benjamin Kehlet 2012
#
# First added:  2008-07-11
# Last changed: 2012-11-12

from dolfin import *

if MPI.num_processes() == 1:
    mesh = UnitIntervalMesh(10)
    print "Plotting a UnitInterval"
    plot(mesh, title="Unit interval")

mesh = UnitSquareMesh(10, 10)
print "Plotting a UnitSquare"
plot(mesh, title="Unit square")

mesh = UnitSquareMesh(10, 10, "left")
print "Plotting a UnitSquare"
plot(mesh, title="Unit square (left)")

mesh = UnitSquareMesh(10, 10, "crossed")
print "Plotting a UnitSquare"
plot(mesh, title="Unit square (crossed)")

mesh = UnitSquareMesh(10, 10, "right/left")
print "Plotting a UnitSquare"
plot(mesh, title="Unit square (right/left)")

mesh = RectangleMesh(0.0, 0.0, 10.0, 4.0, 10, 10)
print "Plotting a Rectangle"
plot(mesh, title="Rectangle")

mesh = RectangleMesh(-3.0, 2.0, 7.0, 6.0, 10, 10, "right/left")
print "Plotting a Rectangle"
plot(mesh, title="Rectangle (right/left)")

mesh = UnitCircleMesh(20, "right", "rotsumn")
print "Plotting a UnitCircle"
plot(mesh, title="Unit circle (rotsum)")

#mesh = UnitCircleMesh(20, "left", "sumn")
#print "Plotting a UnitCircle"
#plot(mesh, title="Unit circle (sumn)")

mesh = UnitCircleMesh(20, "right", "maxn")
print "Plotting a UnitCircle"
plot(mesh, title="Unit circle (maxn)")

mesh = UnitCubeMesh(10, 10, 10)
print "Plotting a UnitCube"
plot(mesh, title="Unit cube")

mesh = BoxMesh(0.0, 0.0, 0.0, 10.0, 4.0, 2.0, 10, 10, 10)
print "Plotting a Box"
plot(mesh, title="Box")

mesh = UnitSphereMesh(10)
print "Plotting a UnitSphere"
plot(mesh, title="Unit sphere")

interactive()
