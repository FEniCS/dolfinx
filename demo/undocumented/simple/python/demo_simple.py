"""This demo program illustrates how to create simple finite element
matrices like the stiffness matrix and mass matrix."""

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
# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008
#
# First added:  2007-11-15
# Last changed: 2008-12-13

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
A0 = uBLASDenseMatrix(4, 4)
M0 = uBLASDenseMatrix(4, 4)
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
info(A, True)
print ""

print "Reference stiffness matrix:"
info(A0, True)
print ""

print "Assembled mass matrix:"
info(M, True)
print ""

print "Reference mass matrix:"
info(M0, True)
print ""
