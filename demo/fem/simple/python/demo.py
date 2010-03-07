"""This demo program illustrates how to create simple finite element
matrices like the stiffness matrix and mass matrix."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2008-12-13"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008

from dolfin import *
from numpy import array

not_working_in_parallel("This demo")

# Load reference mesh (just a simple tetrahedron)
mesh = Mesh("../tetrahedron.xml.gz");

# Assemble stiffness and mass matrices
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
u = TrialFunction(V)
A = assemble(dot(grad(v), grad(u))*dx)
M = assemble(v*u*dx)

# Create reference matrices and set entries
A0 = Matrix(4, 4)
M0 = Matrix(4, 4)
pos = array([0, 1, 2, 3], dtype='I')
A0.set(array([[1.0/2.0, -1.0/6.0, -1.0/6.0, -1.0/6.0],
              [-1.0/6.0, 1.0/6.0, 0.0, 0.0],
              [-1.0/6.0, 0.0, 1.0/6.0, 0.0],
              [-1.0/6.0, 0.0, 0.0, 1.0/6.0]]), pos, pos)

M0.set(array([[1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0],
              [1.0/120.0, 1.0/60.0, 1.0/120.0, 1.0/120.0],
              [1.0/120.0, 1.0/120.0, 1.0/60.0, 1.0/120.0],
              [1.0/120.0, 1.0/120.0, 1.0/120.0, 1.0/60.0]]), pos, pos)
A0.apply("insert")
M0.apply("insert")

# Display matrices
print ""
print "Assembled stiffness matrix:"
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
