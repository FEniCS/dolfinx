# This demo program dmonstrates how to create simple finite
# element matrices like the stiffness matrix and mass matrix.
# For general forms and matrices, forms must be defined and
# compiled with FFC.
#
# Original implementation: ../cpp/main.cpp by Johan Hoffman and Anders Logg.
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
import numpy

#
# THIS DEMO IS CURRENTLY NOT WORKING, SEE NOTE IN CODE.
#

# Load reference mesh (just a simple tetrahedron)
mesh = Mesh("../tetrahedron.xml.gz");

# Create stiffness and mass matrices
# (This doesn't work because StiffnessMatrix.h and MassMatrix.h are not included
# in src/pydolfin/dolfin_headers.h due to ublas issues)
#A = StiffnessMatrix(mesh)
#M = MassMatrix(mesh)

# The following few lines should be deleted once StiffnessMatrix etc. are available
element = FiniteElement("Lagrange", "tetrahedron", 1)
v = TestFunction(element)
u = TrialFunction(element)
stiff = dot(grad(v), grad(u))*dx
mass  = v*u*dx

A = assemble(stiff, mesh)
M = assemble(mass, mesh)

# Create reference matrices
A0_array = numpy.array([[1.0/2.0, -1.0/6.0, -1.0/6.0, -1.0/6.0],
            [-1.0/6.0, 1.0/6.0, 0.0, 0.0],
            [-1.0/6.0, 0.0, 1.0/6.0, 0.0],
            [-1.0/6.0, 0.0, 0.0, 1.0/6.0]])

M0_array = numpy.array([[1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0],
            [1.0/120.0, 1.0/60.0, 1.0/120.0, 1.0/120.0],
            [1.0/120.0, 1.0/120.0, 1.0/60.0, 1.0/120.0],
            [1.0/120.0, 1.0/120.0, 1.0/120.0, 1.0/60.0]])

#   unsigned int position[4] = {0, 1, 2, 3};
position = numpy.array([0, 1, 2, 3], 'uint')

A0 = Matrix(4,4)
M0 = Matrix(4,4)
A0.set(A0_array, 4, position, 4, position)
M0.set(M0_array, 4, position, 4, position)

A0.apply()
M0.apply()

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




