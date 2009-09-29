"""This demo illustrates the built-in mesh types."""

__author__    = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__      = "2008-07-11 -- 2008-12-07"
__copyright__ = "Copyright (C) 2008 Garth N. Wells"
__license__   = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

mesh = UnitInterval(10)
print "Plotting a UnitInterval"
plot(mesh, title="Unit interval")

mesh = UnitSquare(10, 10)
print "Plotting a UnitSquare"
plot(mesh, title="Unit square")

mesh = UnitSquare(10, 10, "left")
print "Plotting a UnitSquare"
plot(mesh, title="Unit square (left)")

mesh = UnitSquare(10, 10, "crossed")
print "Plotting a UnitSquare"
plot(mesh, title="Unit square (crossed)")

mesh = UnitSquare(10, 10, "right/left")
print "Plotting a UnitSquare"
plot(mesh, title="Unit square (right/left)")

mesh = Rectangle(0.0, 0.0, 10.0, 4.0, 10, 10)
print "Plotting a Rectangle"
plot(mesh, title="Rectangle")

mesh = Rectangle(-3.0, 2.0, 7.0, 6.0, 10, 10, "right/left")
print "Plotting a Rectangle"
plot(mesh, title="Rectangle (right/left)")

mesh = UnitCircle(20, "right", "rotsumn")
print "Plotting a UnitCircle"
plot(mesh, title="Unit circle (rotsum)")

#mesh = UnitCircle(20, "left", "sumn")
#print "Plotting a UnitCircle"
#plot(mesh, title="Unit circle (sumn)")

mesh = UnitCircle(20, "right", "maxn")
print "Plotting a UnitCircle"
plot(mesh, title="Unit circle (maxn)")

mesh = UnitCube(10, 10, 10)
print "Plotting a UnitCube"
plot(mesh, title="Unit cube")

mesh = Box(0.0, 0.0, 0.0, 10.0, 4.0, 2.0, 10, 10, 10)
print "Plotting a Box"
plot(mesh, title="Box")

mesh = UnitSphere(10)
print "Plotting a UnitSphere"
plot(mesh, title="Unit sphere")

interactive()
