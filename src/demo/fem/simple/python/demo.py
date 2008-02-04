# This demo program dmonstrates how to create simple finite
# element matrices like the stiffness matrix and mass matrix.
#
# Modified by Anders Logg, 2008

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2008-01-15"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
import numpy

# Load reference mesh (just a simple tetrahedron)
mesh = Mesh("../tetrahedron.xml.gz");

# Assemble stiffness and mass matrices
element = FiniteElement("Lagrange", "tetrahedron", 1)
v = TestFunction(element)
u = TrialFunction(element)
A = assemble(dot(grad(v), grad(u))*dx, mesh)
M = assemble(v*u*dx, mesh)

# Create reference matrices and set entries
A0 = Matrix(4, 4)
M0 = Matrix(4, 4)
pos = numpy.array([0, 1, 2, 3], dtype='I')
A0.set(numpy.array([[1.0/2.0, -1.0/6.0, -1.0/6.0, -1.0/6.0],
                    [-1.0/6.0, 1.0/6.0, 0.0, 0.0],
                    [-1.0/6.0, 0.0, 1.0/6.0, 0.0],
                    [-1.0/6.0, 0.0, 0.0, 1.0/6.0]]), pos, pos)

M0.set(numpy.array([[1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0],
                    [1.0/120.0, 1.0/60.0, 1.0/120.0, 1.0/120.0],
                    [1.0/120.0, 1.0/120.0, 1.0/60.0, 1.0/120.0],
                    [1.0/120.0, 1.0/120.0, 1.0/120.0, 1.0/60.0]]), pos, pos)
A0.apply()
M0.apply()

# Display matrices
print ""
print "Assembled stiffness matrix:"
print A
A.disp()
print ""

print "Reference stiffness matrix:"
A0.disp()
print ""

print "Assembled mass matrix:"
M.disp()
print ""

print "Reference mass matrix:"
M0.disp()
print ""
