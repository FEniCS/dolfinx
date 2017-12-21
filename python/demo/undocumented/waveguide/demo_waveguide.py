""" This demo demonstrates the calculation and visualization of a TM
(Transverse Magnetic) cutoff mode of a rectangular waveguide.

For more information regarding waveguides see

http://www.ee.bilkent.edu.tr/~microwave/programs/magnetic/rect/info.htm

See the pdf in the parent folder and the following reference

The Finite Element in Electromagnetics (2nd Ed)
Jianming Jin [7.2.1 - 7.2.2]

"""
# Copyright (C) 2008 Evan Lezar
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
# Modified by Anders Logg 2008, 2015
#
# First added:  2008-08-22
# Last changed: 2015-06-15


from dolfin import *

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# Make sure we use the PETSc backend
parameters["linear_algebra_backend"] = "PETSc"

# Create mesh
width = 1.0
height = 0.5
mesh = RectangleMesh(Point(0, 0), Point(width, height), 4, 2)

# Define the function space
V = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 3)

# Define the test and trial functions
v = TestFunction(V)
u = TrialFunction(V)

# Define the forms - generates an generalized eigenproblem of the form
# [S]{h} = k_o^2[T]{h}
# with the eigenvalues k_o^2 representing the square of the cutoff wavenumber
# and the corresponding right-eigenvector giving the coefficients of the
# discrete system used to obtain the approximate field anywhere in the domain
def curl_t(w):
    return Dx(w[1], 0) - Dx(w[0], 1)
s = curl_t(v)*curl_t(u)*dx
t = inner(v, u)*dx

# Assemble the stiffness matrix (S) and mass matrix (T)
S = PETScMatrix()
T = PETScMatrix()
assemble(s, tensor=S)
assemble(t, tensor=T)

# Solve the eigensystem
esolver = SLEPcEigenSolver(S, T)
esolver.parameters["spectrum"] = "smallest real"
esolver.parameters["solver"] = "lapack"
esolver.solve()

# The result should have real eigenvalues but due to rounding errors, some of
# the resultant eigenvalues may be small complex values.
# only consider the real part

# Now, the system contains a number of zero eigenvalues (near zero due to
# rounding) which are eigenvalues corresponding to the null-space of the curl
# operator and are a mathematical construct and do not represent physically
# realizable modes.  These are called spurious modes.
# So, we need to identify the smallest, non-zero eigenvalue of the system -
# which corresponds with cutoff wavenumber of the the dominant cutoff mode.
cutoff = None
for i in range(S.size(1)):
    (lr, lc) = esolver.get_eigenvalue(i)
    if lr > 1 and lc == 0:
        cutoff = sqrt(lr)
        break

if cutoff is None:
    print("Unable to find dominant mode")
else:
    print("Cutoff frequency:", cutoff)
