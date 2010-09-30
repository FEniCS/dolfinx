__author__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2010 Marie E. Rognes"
__license__  = "GNU LGPL Version 3.0 or any later version"

from dolfin import *

# Test for PETSc and SLEPc
if not has_la_backend("PETSc"):
    print "DOLFIN has not been configured with PETSc. Exiting."
    exit()

if not has_slepc():
    print "DOLFIN has not been configured with SLEPc. Exiting."
    exit()

# Make sure we use the PETSc backend
parameters["linear_algebra_backend"] = "PETSc"

# Create mesh
n = 32
mesh = UnitSquare(n, n)

# Define the function space
V = FunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)
W = V * Q

(p, u) = TestFunctions(W)
(q, v) = TrialFunctions(W)

a = (dot(grad(p), v) + dot(u, grad(q)) + curl(u)*curl(v))*dx
b = (p*q + dot(u, v))*dx

A = PETScMatrix()
B = PETScMatrix()

assemble(a, tensor=A)
assemble(b, tensor=B)

esolver = SLEPcEigenSolver()

# Find values closest in magnitude to '10.0'
esolver.parameters["spectral_transform"] =  "shift-and-invert"
esolver.parameters["spectrum"] = "target magnitude"
esolver.parameters["spectral_shift"] =  10.0

# Solve the eigensystem
esolver.solve(A, B, 1)

m = esolver.get_number_converged()
print "m = ", m

for i in range(m):
    (lr, lc) = esolver.get_eigenvalue(i)
    print "lambda_%d = " % i, (lr, lc)

