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
# Begin demo

from dolfin import *

if MPI.num_processes() == 1:
    mesh = UnitIntervalMesh(10)
    print "Plotting a UnitIntervalMesh"
    plot(mesh, title="Unit interval")

mesh = UnitSquareMesh(10, 10)
print "Plotting a UnitSquareMesh"
p=plot(mesh, title="Unit square")
p.write_png("unitsquaremesh")

mesh = UnitSquareMesh(10, 10, "left")
print "Plotting a UnitSquareMesh"
p=plot(mesh, title="Unit square (left)")
p.write_png("unitsquaremesh_left")

mesh = UnitSquareMesh(10, 10, "crossed")
print "Plotting a UnitSquareMesh"
p=plot(mesh, title="Unit square (crossed)")
p.write_png("unitsquaremesh_crossed")

mesh = UnitSquareMesh(10, 10, "right/left")
print "Plotting a UnitSquareMesh"
p=plot(mesh, title="Unit square (right/left)")
p.write_png("unitsquaremesh_left_right")

mesh = RectangleMesh(0.0, 0.0, 10.0, 4.0, 10, 10)
print "Plotting a RectangleMesh"
p=plot(mesh, title="Rectangle")
p.write_png("rectanglemesh")

mesh = RectangleMesh(-3.0, 2.0, 7.0, 6.0, 10, 10, "right/left")
print "Plotting a RectangleMesh"
p=plot(mesh, title="Rectangle (right/left)")
p.write_png("rectanglemesh_left_right")

mesh = UnitCircleMesh(20, "right", "rotsumn")
print "Plotting a UnitCircleMesh"
p=plot(mesh, title="Unit circle (rotsum)")
p.write_png("unitcirclemesh_rotsum")

#mesh = UnitCircleMesh(20, "left", "sumn")
#print "Plotting a UnitCircle"
#plot(mesh, title="Unit circle (sumn)")

mesh = UnitCircleMesh(20, "right", "maxn")
print "Plotting a UnitCircleMesh"
p=plot(mesh, title="Unit circle (maxn)")
p.write_png("unitcirclemesh_maxn")

mesh = UnitCubeMesh(10, 10, 10)
print "Plotting a UnitCubeMesh"
p=plot(mesh, title="Unit cube")
p.write_png("unitcubemesh")

mesh = BoxMesh(0.0, 0.0, 0.0, 10.0, 4.0, 2.0, 10, 10, 10)
print "Plotting a BoxMesh"
p=plot(mesh, title="Box")
p.write_png("boxmesh")

interactive()
