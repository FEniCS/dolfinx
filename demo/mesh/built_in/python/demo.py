# The demo illustrates the built-in mesh types.

__author__    = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__      = "2008-07-11 -- 2008-07-14"
__copyright__ = "Copyright (C) 2008 Garth N. Wells"
__license__   = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Create and plot built-in meshes

mesh = UnitInterval(10)
print "Plotting a UnitInterval"
plot(mesh)

mesh = UnitSquare(10, 10)
print "Plotting a UnitSquare"
plot(mesh)

mesh = Rectangle(0.0, 10.0, 0.0, 4.0, 10, 10)
print "Plotting a Rectangle"
plot(mesh)

mesh = UnitCircle(20, UnitCircle.right)
print "Plotting a UnitCircle"
plot(mesh)

mesh = UnitCube(10, 10, 10)
print "Plotting a UnitCube"
plot(mesh)

mesh = Box(0.0, 10.0, 0.0, 4.0, 0.0, 2.0, 10, 10, 10)
print "Plotting a Box"
plot(mesh)

interactive()
