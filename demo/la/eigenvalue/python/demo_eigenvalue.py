"""
This program illustrates basic use of the SLEPc eigenvalue solver for
a standard eigenvalue problem.
"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2009-10-09"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.
# Modified by Marie Rognes, 2009.

# Begin demo

from dolfin import *

# Test for PETSc and SLEPc
if not has_la_backend("PETSc"):
    print "DOLFIN has not been configured with PETSc. Exiting."
    exit()

if not has_slepc():
    print "DOLFIN has not been configured with SLEPc. Exiting."
    exit()

# Define mesh, function space
mesh = Mesh("box_with_dent.xml.gz")
V = FunctionSpace(mesh, "CG", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx

# Assemble stiffness form
A = PETScMatrix()
assemble(a, tensor=A)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print "Computing eigenvalues. This can take a minute."
eigensolver.solve()

# Extract largest (first) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(0)

print "Largest eigenvalue: ", r

# Initialize function with eigenvector
u = Function(V, rx)

# Plot eigenfunction
plot(u)
interactive()
